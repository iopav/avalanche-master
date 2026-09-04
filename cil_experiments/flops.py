from __future__ import annotations

import contextlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode


PHASES = (
    "core_training",
    "learning_auxiliary",
    "single_sample_forward",
)

CORE_TRAINING_PHASES = ("core_training",)
LEARNING_AUXILIARY_PHASES = ("learning_auxiliary",)


def _is_shape(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(isinstance(v, int) for v in value)


def _numel(shape: Any) -> int:
    if shape is None:
        return 0
    if _is_shape(shape):
        return math.prod(shape)
    if isinstance(shape, (list, tuple)):
        return sum(_numel(v) for v in shape)
    return 0


def _output_numel(out_shape: Any, fallback: Any = None) -> int:
    value = _numel(out_shape)
    return value if value else _numel(fallback)


def unary_flop(input_shape, *args, out_shape=None, **kwargs) -> int:
    return _output_numel(out_shape, input_shape)


def binary_flop(input_shape, other_shape=None, *args, out_shape=None, **kwargs) -> int:
    return _output_numel(out_shape, input_shape)


def add_like_flop(input_shape, other_shape=None, *args, out_shape=None, **kwargs) -> int:
    elements = _output_numel(out_shape, input_shape)
    # Plain a +/- b is one FLOP. A non-default scaled overload
    # a +/- alpha*b executes one multiply and one add/sub. PyTorch canonicalizes
    # an explicit alpha=1 to the ordinary overload, for which no multiply is
    # executed by the dispatched arithmetic operator.
    return elements * (2 if "alpha" in kwargs else 1)


def reduction_flop(input_shape, *args, out_shape=None, **kwargs) -> int:
    n = _numel(input_shape)
    out = max(1, _numel(out_shape))
    return max(0, n - out)


def mean_flop(input_shape, *args, out_shape=None, **kwargs) -> int:
    n = _numel(input_shape)
    out = max(1, _numel(out_shape))
    return max(0, n - out) + out


def variance_flop(input_shape, *args, out_shape=None, **kwargs) -> int:
    """Dense two-pass variance: mean, center, square, reduce and divide."""
    n = _numel(input_shape)
    # Per group of k values: mean k, center k, square k, final reduce/divide k.
    return 4 * n


def norm_flop(input_shape, *args, out_shape=None, **kwargs) -> int:
    # square + accumulation + square root
    n = _numel(input_shape)
    out = max(1, _numel(out_shape))
    return n + max(0, n - out) + out


def group_norm_flop(input_shape, *args, out_shape=None, **kwargs) -> int:
    # mean, variance, normalization, affine scale and shift.
    return 7 * _numel(input_shape)


def group_norm_backward_flop(grad_shape, input_shape, *args, out_shape=None, **kwargs) -> int:
    return 15 * _numel(input_shape)


def batch_norm_flop(
    input_shape, weight_shape=None, bias_shape=None, running_mean_shape=None,
    running_var_shape=None, training=True, *args, out_shape=None, **kwargs
) -> int:
    # Explicit dense approximation. Training computes batch mean/variance,
    # normalization, affine transform and running-stat updates; evaluation only
    # normalizes with stored statistics and applies the affine transform.
    return (8 if bool(training) else 4) * _numel(input_shape)


def batch_norm_backward_flop(
    grad_shape, input_shape, weight_shape=None, running_mean_shape=None,
    running_var_shape=None, save_mean_shape=None, save_invstd_shape=None,
    train=True, eps=1e-5, output_mask=None, *args, out_shape=None, **kwargs
) -> int:
    return 15 * _numel(input_shape)


def batch_norm_eval_flop(input_shape, *args, out_shape=None, **kwargs) -> int:
    return 4 * _numel(input_shape)


def xlogy_flop(input_shape, other_shape=None, *args, out_shape=None, **kwargs) -> int:
    # x * log(y): one logarithm and one multiplication per broadcast output.
    return 2 * _output_numel(out_shape, input_shape)


def _adaptive_pool_contributions(input_shape: Any, output_shape: Any) -> int:
    if not _is_shape(input_shape) or not _is_shape(output_shape):
        return 0
    if len(input_shape) < 3 or len(input_shape) != len(output_shape):
        return 0
    axis_totals = []
    for input_size, output_size in zip(input_shape[2:], output_shape[2:]):
        if output_size <= 0:
            return 0
        axis_totals.append(
            sum(
                math.ceil((index + 1) * input_size / output_size)
                - math.floor(index * input_size / output_size)
                for index in range(output_size)
            )
        )
    return int(input_shape[0] * input_shape[1] * math.prod(axis_totals))


def adaptive_pool_flop(input_shape, *args, out_shape=None, **kwargs) -> int:
    # A bin of k values uses k - 1 additions and one division: k FLOPs.
    return _adaptive_pool_contributions(input_shape, out_shape)


def adaptive_pool_backward_flop(grad_shape, input_shape, *args, out_shape=None, **kwargs) -> int:
    # One division per gradient contribution. Overlapping adaptive bins also
    # need additions after the first contribution received by an input cell.
    contributions = _adaptive_pool_contributions(input_shape, grad_shape)
    return contributions + max(0, contributions - _numel(input_shape))


def _reduction_group_count(shape, args, kwargs) -> int:
    dim = args[0] if args and isinstance(args[0], int) else kwargs.get("dim", -1)
    dim = int(dim) % len(shape)
    return max(1, _numel(shape) // max(1, int(shape[dim])))


def softmax_flop(input_shape, *args, out_shape=None, **kwargs) -> int:
    # Max selection is comparison work. Each reduction group performs k
    # subtractions, k exponentials, k-1 additions and k divisions.
    groups = _reduction_group_count(input_shape, args, kwargs)
    return 4 * _numel(input_shape) - groups


def log_softmax_flop(input_shape, *args, out_shape=None, **kwargs) -> int:
    # Stable log-softmax adds one logarithm per group in place of the softmax
    # division saving, for exactly 4*k FLOPs per group.
    return 4 * _numel(input_shape)


def softmax_backward_flop(grad_shape, output_shape, *args, out_shape=None, **kwargs) -> int:
    # Per reduction group: k products, k-1 additions, k subtractions and k products.
    groups = _reduction_group_count(output_shape, args, kwargs)
    return 4 * _numel(output_shape) - groups


def logsumexp_flop(input_shape, *args, out_shape=None, **kwargs) -> int:
    # Max selection is comparison work under this protocol. For each reduction
    # group: k subtracts + k exponentials + (k-1) additions + one logarithm
    # + one restoration add = 3*k+1 FLOPs.
    groups = max(1, _numel(out_shape))
    return 3 * _numel(input_shape) + groups


def _loss_reduction_flops(elements: int, reduction: int) -> int:
    if reduction == 0 or elements <= 0:  # none
        return 0
    if reduction == 2:  # sum
        return max(0, elements - 1)
    if reduction == 1:  # mean: reduction plus one division
        return elements
    raise ValueError(f"Unknown PyTorch loss reduction code: {reduction}")


def nll_loss_forward_flop(
    input_shape, target_shape, weight_shape, reduction, *args, out_shape=None, **kwargs
) -> int:
    samples = _numel(target_shape)
    selected_value = samples  # negate the selected log-probability
    weight_multiply = samples if weight_shape is not None else 0
    return selected_value + weight_multiply + _loss_reduction_flops(samples, int(reduction))


def nll_loss_backward_flop(
    grad_shape, input_shape, target_shape, weight_shape, reduction, *args, out_shape=None, **kwargs
) -> int:
    samples = _numel(target_shape)
    per_sample = 1  # sign change of grad_output
    if weight_shape is not None:
        per_sample += 1
    if int(reduction) == 1:
        per_sample += 1
    return samples * per_sample


def binary_cross_entropy_flop(
    input_shape, target_shape, weight_shape=None, reduction=1, *args, out_shape=None, **kwargs
) -> int:
    n = _numel(input_shape)
    # 1-x, log(x), log(1-x), 1-y, two multiplies, add, negate.
    elementwise = 8 * n + (n if weight_shape is not None else 0)
    return elementwise + _loss_reduction_flops(n, int(reduction))


def binary_cross_entropy_backward_flop(
    grad_shape, input_shape, target_shape, weight_shape=None, reduction=1, *args, out_shape=None, **kwargs
) -> int:
    n = _numel(input_shape)
    # x-y, 1-x, x*(1-x), division and grad_output scaling.
    per_element = 5 + (1 if weight_shape is not None else 0) + (1 if int(reduction) == 1 else 0)
    return n * per_element


def mse_loss_flop(
    input_shape, target_shape, reduction=1, *args, out_shape=None, **kwargs
) -> int:
    n = _output_numel(input_shape, target_shape)
    return 2 * n + _loss_reduction_flops(n, int(reduction))


def mse_loss_backward_flop(
    grad_shape, input_shape, target_shape, reduction, *args, out_shape=None, **kwargs
) -> int:
    n = _numel(input_shape)
    # input-target, multiply by 2 and multiply by grad_output; mean adds division.
    return n * (3 + (1 if int(reduction) == 1 else 0))


def convolution_flop(
    x_shape,
    w_shape,
    bias_shape,
    stride,
    padding,
    dilation,
    transposed,
    *args,
    out_shape=None,
    **kwargs,
) -> int:
    batch = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    c_out, c_in, *kernel = w_shape
    macs = math.prod(conv_shape) * math.prod(kernel) * batch * c_out * c_in
    bias_adds = _numel(out_shape) if bias_shape is not None else 0
    return 2 * macs + bias_adds


def convolution_backward_flop(
    grad_out_shape,
    x_shape,
    w_shape,
    bias_shape,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
    *args,
    out_shape=None,
    **kwargs,
) -> int:
    """Convolution backward under the same 1 MAC = 2 FLOPs convention.尽管在这里接受了group，但是不支持group>1的情况，无意外情况应该用不到，懒得写了8.25"""

    def transpose_channels(shape):
        return [shape[1], shape[0], *shape[2:]]

    def shape_at(index):
        if isinstance(out_shape, (list, tuple)) and index < len(out_shape):
            candidate = out_shape[index]
            return candidate if _is_shape(candidate) else None
        return None

    def contraction_flops(inp, weight, output, is_transposed):
        if not (_is_shape(inp) and _is_shape(weight) and _is_shape(output)):
            return 0
        conv_shape = (inp if is_transposed else output)[2:]
        c_out, c_in, *kernel = weight
        macs = math.prod(conv_shape) * math.prod(kernel) * inp[0] * c_out * c_in
        return 2 * macs

    total = 0
    if output_mask[0]:
        total += contraction_flops(grad_out_shape, w_shape, shape_at(0), not transposed)
    if output_mask[1]:
        grad_weight_shape = shape_at(1)
        if transposed:
            total += contraction_flops(
                transpose_channels(grad_out_shape),
                transpose_channels(x_shape),
                transpose_channels(grad_weight_shape) if grad_weight_shape else None,
                False,
            )
        else:
            total += contraction_flops(
                transpose_channels(x_shape),
                transpose_channels(grad_out_shape),
                transpose_channels(grad_weight_shape) if grad_weight_shape else None,
                False,
            )
    if output_mask[2] and bias_shape is not None:
        channels = int(bias_shape[0])
        values_per_channel = _numel(grad_out_shape) // channels
        total += channels * max(0, values_per_channel - 1)
    return int(total)


def addmm_flop(self_shape, a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
    m, k = a_shape
    _, n = b_shape
    return 2 * m * n * k + m * n


def baddbmm_flop(self_shape, a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
    batch, m, k = a_shape
    _, _, n = b_shape
    return 2 * batch * m * n * k + batch * m * n


def covariance_flop(input_shape, *args, out_shape=None, **kwargs) -> int:
    # torch.cov receives variables x observations. Center + X @ X.T + division.
    if len(input_shape) != 2:
        return 0
    variables, observations = input_shape
    center = variables * observations * 2
    matmul = 2 * variables * variables * observations
    divide = variables * variables
    return center + matmul + divide


def svd_flop(input_shape, *args, out_shape=None, **kwargs) -> int:
    if len(input_shape) < 2:
        return 0
    m, n = input_shape[-2:]
    if m < n:
        m, n = n, m
    # Golub-Reinsch dense SVD convention; explicitly approximate.
    return int(4 * m * n * n + (8.0 / 3.0) * n**3)


def pinv_flop(input_shape, *args, out_shape=None, **kwargs) -> int:
    if len(input_shape) < 2:
        return 0
    m, n = input_shape[-2:]
    return svd_flop(input_shape) + 2 * m * n * min(m, n)


def euclidean_distance_flop(x_shape, y_shape, *args, out_shape=None, **kwargs) -> int:
    if len(x_shape) < 2 or len(y_shape) < 2:
        return 0
    d = x_shape[-1]
    if y_shape[-1] != d:
        return 0
    pairs = _output_numel(out_shape)
    if pairs == 0:
        batch = math.prod(x_shape[:-2]) if len(x_shape) > 2 else 1
        pairs = batch * x_shape[-2] * y_shape[-2]
    # Per pair: d subtracts, d squares, d-1 additions and one square root.
    return 3 * pairs * d


def build_custom_mapping() -> dict[Any, Callable[..., int]]:
    aten = torch.ops.aten
    mapping: dict[Any, Callable[..., int]] = {}

    def add(name: str, formula: Callable[..., int]):
        op = getattr(aten, name, None)
        if op is not None:
            mapping[op] = formula

    for name in ("add", "add_", "sub", "sub_"):
        add(name, add_like_flop)
    add("rsub", binary_flop)
    for name in ("mul", "mul_", "div", "div_", "pow", "pow_", "exp", "exp_", "log", "log_", "sqrt", "sqrt_", "rsqrt", "reciprocal", "sigmoid", "sigmoid_backward"):
        add(name, unary_flop if name not in {"mul", "mul_", "div", "div_"} else binary_flop)
    add("neg", unary_flop)
    for name in ("sum", "prod"):
        add(name, reduction_flop)
    add("mean", mean_flop)
    add("var", variance_flop)
    add("var_mean", variance_flop)
    for name in ("linalg_vector_norm", "norm"):
        add(name, norm_flop)
    add("native_group_norm", group_norm_flop)
    add("native_group_norm_backward", group_norm_backward_flop)
    add("native_batch_norm", batch_norm_flop)
    add("_native_batch_norm_legit", batch_norm_flop)
    add("_native_batch_norm_legit_functional", batch_norm_flop)
    add("_native_batch_norm_legit_no_training", batch_norm_eval_flop)
    add("native_batch_norm_backward", batch_norm_backward_flop)
    add("native_layer_norm", group_norm_flop)
    add("native_layer_norm_backward", group_norm_backward_flop)
    add("xlogy", xlogy_flop)
    add("adaptive_avg_pool1d", adaptive_pool_flop)
    add("adaptive_avg_pool2d", adaptive_pool_flop)
    add("_adaptive_avg_pool2d_backward", adaptive_pool_backward_flop)
    add("_softmax", softmax_flop)
    add("_log_softmax", log_softmax_flop)
    add("_softmax_backward_data", softmax_backward_flop)
    add("_log_softmax_backward_data", softmax_backward_flop)
    add("logsumexp", logsumexp_flop)
    add("nll_loss_forward", nll_loss_forward_flop)
    add("nll_loss_backward", nll_loss_backward_flop)
    add("binary_cross_entropy", binary_cross_entropy_flop)
    add("binary_cross_entropy_backward", binary_cross_entropy_backward_flop)
    add("mse_loss", mse_loss_flop)
    add("mse_loss_backward", mse_loss_backward_flop)
    add("convolution", convolution_flop)
    add("_convolution", convolution_flop)
    add("cudnn_convolution", convolution_flop)
    add("convolution_overrideable", convolution_flop)
    add("convolution_backward", convolution_backward_flop)
    add("addmm", addmm_flop)
    add("baddbmm", baddbmm_flop)
    add("cov", covariance_flop)
    add("_linalg_svd", svd_flop)
    add("linalg_pinv", pinv_flop)
    add("_cdist_forward", euclidean_distance_flop)
    add("_euclidean_dist", euclidean_distance_flop)
    return mapping


ZERO_FLOP_OPERATIONS = {
    "aten.alias", "aten.as_strided", "aten.cat", "aten.clone", "aten.contiguous",
    "aten.copy_", "aten.detach", "aten.empty", "aten.expand", "aten.fill_",
    "aten.flatten", "aten.gather", "aten.index", "aten.index_put", "aten.index_put_", "aten.isfinite", "aten.item",
    "aten.lift_fresh", "aten.masked_fill", "aten.masked_fill_", "aten.new_", "aten.ones", "aten.permute",
    "aten.repeat", "aten.reshape", "aten.roll", "aten.scatter", "aten.scatter_", "aten.select", "aten.slice", "aten.slice_backward",
    "aten.split", "aten.squeeze", "aten.stack", "aten.t", "aten.to", "aten.transpose",
    "aten.unbind", "aten.unsqueeze", "aten.view", "aten.where", "aten.zero_", "aten.zeros",
    "aten._local_scalar_dense", "aten._to_copy", "aten.arange", "aten.scalar_tensor",
    "aten.full", "aten.full_like", "aten.empty_like", "aten.ones_like", "aten.zeros_like",
    "aten.rand", "aten.random_", "aten.randperm", "aten.bernoulli", "aten.uniform_", "aten.normal_",
    "aten.eq", "aten.ne", "aten.equal", "aten.lt", "aten.le", "aten.gt", "aten.ge",
    "aten.argmax", "aten.argmin", "aten.sort", "aten.topk",
    "aten.unique", "aten._unique", "aten.any", "aten.all", "aten.nonzero",
    "aten.threshold", "aten.relu", "aten.relu_", "aten.hardtanh", "aten.dropout", "aten.native_dropout",
    "aten.pin_memory", "aten.record_stream", "aten.is_contiguous",
    "aten.size", "aten.stride", "aten.numel", "aten.dim", "prim.device",
    "aten._pin_memory", "aten.is_pinned", "aten.max", "aten.min", "aten.clamp",
    "aten.clamp_min", "aten.clamp_min_", "aten.clamp_max", "aten.clamp_max_",
    "aten.set_", "aten.diagonal", "aten.eye", "aten._unsafe_view",
}

ZERO_FLOP_PREFIXES = (
    "aten.new_",
    "aten.logical",
    "aten.bitwise",
    "aten._unique",
    "aten.threshold",
    "aten.sym_",
    "aten.max",
    "aten.min",
    "profiler._record_function_enter",
    "profiler._record_function_exit",
)


def is_explicit_zero_flop(packet: Any) -> bool:
    name = str(packet)
    return name in ZERO_FLOP_OPERATIONS or name.startswith(ZERO_FLOP_PREFIXES)


def _tensor_signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, (tuple, list)):
        return [_tensor_signature(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _tensor_signature(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


class LeafOperatorAuditMode(TorchDispatchMode):
    def __init__(self, phase_getter: Callable[[], str]):
        super().__init__()
        self.phase_getter = phase_getter
        self.calls: dict[str, Counter[Any]] = defaultdict(Counter)
        self.examples: dict[Any, Any] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        packet = func._overloadpacket if hasattr(func, "_overloadpacket") else func
        self.calls[self.phase_getter()][packet] += 1
        self.examples.setdefault(packet, {"args": _tensor_signature(args), "kwargs": _tensor_signature(kwargs)})
        return func(*args, **kwargs)


@dataclass
class ExperienceFlopResult:
    total_flops: int
    phase_flops: dict[str, int]
    terminal_epoch_flops: int
    terminal_epoch_samples: int
    terminal_flops_per_sample: float
    operation_calls: dict[str, dict[str, int]]
    zero_flop_operations: dict[str, int]
    custom_formulas: list[str]
    manual_supplementary_rules: list[dict[str, Any]]
    manual_nonflop_operations: list[dict[str, Any]]


def summarize_learning_flops(
    experience_results: list[ExperienceFlopResult],
    single_sample_forward_flops: int,
) -> dict[str, int]:
    if single_sample_forward_flops < 0:
        raise ValueError("FLOP summary inputs must be non-negative")
    phase_totals: Counter[str] = Counter()
    recorded_training_total = 0
    for result in experience_results:
        phase_totals.update(result.phase_flops)
        recorded_training_total += int(result.total_flops)
    unexpected = set(phase_totals) - set(CORE_TRAINING_PHASES) - set(LEARNING_AUXILIARY_PHASES)
    if unexpected:
        raise AssertionError(f"Training results contain non-learning FLOP phases: {sorted(unexpected)}")
    core = int(sum(phase_totals[phase] for phase in CORE_TRAINING_PHASES))
    auxiliary = int(sum(phase_totals[phase] for phase in LEARNING_AUXILIARY_PHASES))
    if core + auxiliary != recorded_training_total:
        raise AssertionError(
            "Core/auxiliary phase sum differs from recorded cumulative training FLOPs: "
            f"{core} + {auxiliary} != {recorded_training_total}"
        )
    return {
        "overall_learning_flops": core + auxiliary,
        "core_training_flops": core,
        "learning_auxiliary_flops": auxiliary,
        "single_sample_forward_flops": int(single_sample_forward_flops),
    }


def summarize_auxiliary_nonflop_ops(
    experience_results: list[ExperienceFlopResult],
) -> dict[str, Any]:
    calls: Counter[str] = Counter()
    manual_operations: list[dict[str, Any]] = []
    for result in experience_results:
        calls.update(result.zero_flop_operations)
        manual_operations.extend(result.manual_nonflop_operations)
    ordered_calls = {name: int(count) for name, count in sorted(calls.items())}
    return {
        "explicit_nonflop_operator_calls": ordered_calls,
        "manual_nonflop_operations": manual_operations,
        "total_recorded_events": int(
            sum(ordered_calls.values())
            + sum(int(operation["calls"]) for operation in manual_operations)
        ),
        "policy": (
            "Only comparisons, indexing, shape/view, data movement, integer/bit operations and "
            "other explicitly documented non-floating-point operations are listed here; unknown "
            "computational operators still abort the run."
        ),
    }


class PhaseFlopProfiler:
    def __init__(self):
        self.phase = "learning_auxiliary"
        self.custom_mapping = build_custom_mapping()
        self.counter = FlopCounterMode(display=False, custom_mapping=self.custom_mapping)
        self.audit = LeafOperatorAuditMode(lambda: self.phase)
        self.phase_flops: Counter[str] = Counter()
        self.manual_phase_flops: Counter[str] = Counter()
        self.manual_rules: list[dict[str, Any]] = []
        self.manual_nonflop_operations: list[dict[str, Any]] = []
        self._last_total = 0
        self._started = False
        self._epoch_start_total = 0
        self._epoch_processed = 0
        self._last_epoch_flops = 0
        self._last_epoch_samples = 0
        self._last_epoch_end_total = 0

    def start(self) -> None:
        if self._started:
            raise RuntimeError("FLOP profiler is not re-entrant")
        self.audit.__enter__()
        self.counter.__enter__()
        self._started = True
        self._last_total = self.counter.get_total_flops()

    def _combined_total(self) -> int:
        return int(self.counter.get_total_flops() + sum(self.manual_phase_flops.values()))

    def switch(self, phase: str) -> None:
        if phase not in PHASES:
            raise ValueError(f"Unknown FLOP phase: {phase}")
        if self._started:
            current = self.counter.get_total_flops()
            self.phase_flops[self.phase] += int(current - self._last_total)
            self._last_total = current
        self.phase = phase

    @contextlib.contextmanager
    def temporary_phase(self, phase: str) -> Iterator[None]:
        previous = self.phase
        self.switch(phase)
        try:
            yield
        finally:
            self.switch(previous)

    def begin_epoch(self) -> None:
        self.switch("learning_auxiliary")
        self._epoch_start_total = self._combined_total()
        self._epoch_processed = 0

    def add_processed_samples(self, count: int) -> None:
        self._epoch_processed += int(count)

    def end_epoch(self) -> None:
        self.switch("learning_auxiliary")
        self._last_epoch_flops = self._combined_total() - self._epoch_start_total
        self._last_epoch_samples = self._epoch_processed
        self._last_epoch_end_total = self._combined_total()

    def add_manual(self, phase: str, flops: int, formula: str, variables: dict[str, Any]) -> None:
        if flops < 0:
            raise ValueError("Manual FLOPs cannot be negative")
        self.manual_phase_flops[phase] += int(flops)
        self.manual_rules.append({"phase": phase, "flops": int(flops), "formula": formula, "variables": variables})

    def add_nonflop(
        self,
        name: str,
        calls: int,
        reason: str,
        variables: dict[str, Any] | None = None,
    ) -> None:
        if not name or calls < 0 or not reason:
            raise ValueError("Manual non-FLOP records require name, non-negative calls and reason")
        self.manual_nonflop_operations.append(
            {
                "name": name,
                "calls": int(calls),
                "reason": reason,
                "variables": dict(variables or {}),
            }
        )

    def stop(self, strict: bool = True) -> ExperienceFlopResult:
        if not self._started:
            raise RuntimeError("FLOP profiler was not started")
        self.switch("learning_auxiliary")
        counter_registry = set(self.counter.flop_registry)
        self.counter.__exit__(None, None, None)
        self.audit.__exit__(None, None, None)
        self._started = False
        unknown = {}
        zero_calls: Counter[str] = Counter()
        for phase, calls in self.audit.calls.items():
            for packet, count in calls.items():
                if packet in counter_registry:
                    continue
                if is_explicit_zero_flop(packet):
                    zero_calls[str(packet)] += count
                else:
                    unknown[str(packet)] = {
                        "calls": count,
                        "phase": phase,
                        "example": self.audit.examples.get(packet),
                    }
        if strict and unknown:
            formatted = "\n".join(f"{name}: {details}" for name, details in sorted(unknown.items()))
            raise RuntimeError(f"Unsupported computational operators were executed:\n{formatted}")
        for phase, value in self.manual_phase_flops.items():
            self.phase_flops[phase] += value
        total = int(sum(self.phase_flops.values()))
        post_epoch = total - self._last_epoch_end_total
        terminal = self._last_epoch_flops + max(0, post_epoch) #最后一个epoch之后还有额外的flops需要被计入8.29
        if self._last_epoch_samples <= 0:
            raise RuntimeError("Terminal epoch processed-sample count is zero")
        calls_json = {
            phase: {str(packet): int(count) for packet, count in sorted(calls.items(), key=lambda item: str(item[0]))}
            for phase, calls in self.audit.calls.items()
        }
        return ExperienceFlopResult(
            total_flops=total,
            phase_flops={phase: int(self.phase_flops.get(phase, 0)) for phase in PHASES if self.phase_flops.get(phase, 0)},
            terminal_epoch_flops=int(terminal),
            terminal_epoch_samples=int(self._last_epoch_samples),
            terminal_flops_per_sample=float(terminal / self._last_epoch_samples),
            operation_calls=calls_json,
            zero_flop_operations=dict(zero_calls),
            custom_formulas=sorted(str(op) for op in self.custom_mapping),
            manual_supplementary_rules=list(self.manual_rules),
            manual_nonflop_operations=list(self.manual_nonflop_operations),
        )

    def abort(self) -> None:
        if not self._started:
            return
        self.counter.__exit__(None, None, None)
        self.audit.__exit__(None, None, None)
        self._started = False


class FlopPhasePlugin:
    """Avalanche plugin that labels operations while one global FlopCounterMode is active."""

    def __init__(self):
        from avalanche.core import SupervisedPlugin

        if not isinstance(self, SupervisedPlugin):
            # The dynamic base class below is used to keep import-time requirements local.
            pass
        self.profiler: PhaseFlopProfiler | None = None

    def attach(self, profiler: PhaseFlopProfiler) -> None:
        self.profiler = profiler

    def detach(self) -> None:
        self.profiler = None

    def before_training_epoch(self, strategy, **kwargs):
        if self.profiler:
            self.profiler.begin_epoch()

    def after_training_epoch(self, strategy, **kwargs):
        if self.profiler:
            self.profiler.end_epoch()

    def before_training_iteration(self, strategy, **kwargs):
        strategy._cil_extra_processed_samples = 0

    def before_forward(self, strategy, **kwargs):
        if self.profiler:
            self.profiler.switch("core_training")

    def after_forward(self, strategy, **kwargs):
        pass

    def before_backward(self, strategy, **kwargs):
        pass

    def after_backward(self, strategy, **kwargs):
        pass

    def before_update(self, strategy, **kwargs):
        pass

    def after_update(self, strategy, **kwargs):
        if self.profiler:
            self.profiler.switch("learning_auxiliary")

    def after_training_iteration(self, strategy, **kwargs):
        if self.profiler:
            current = int(len(strategy.mb_y))
            extra = int(getattr(strategy, "_cil_extra_processed_samples", 0))
            self.profiler.add_processed_samples(current + extra)


def make_flop_phase_plugin():
    from avalanche.core import SupervisedPlugin

    class _Plugin(FlopPhasePlugin, SupervisedPlugin):
        def __init__(self):
            SupervisedPlugin.__init__(self)
            FlopPhasePlugin.__init__(self)

    return _Plugin()


def profile_single_forward(model, sample: torch.Tensor) -> tuple[int, dict[str, Any]]:
    profiler = PhaseFlopProfiler()
    profiler.phase = "single_sample_forward"
    was_training = model.training
    model.eval()
    try:
        profiler.start()
        with torch.no_grad():
            model(sample)
        # Single-forward profiling has no epoch denominator, so close manually.
        profiler.switch("single_sample_forward")
        registry = set(profiler.counter.flop_registry)
        profiler.counter.__exit__(None, None, None)
        profiler.audit.__exit__(None, None, None)
        profiler._started = False
        unknown = [
            str(op)
            for calls in profiler.audit.calls.values()
            for op in calls
            if op not in registry and not is_explicit_zero_flop(op)
        ]
        if unknown:
            raise RuntimeError(
                f"Unsupported operators in single-sample forward: {sorted(set(unknown))}"
            )
        total = int(sum(profiler.phase_flops.values()))
        detail = {
            "flops": total,
            "operation_calls": {
                phase: {str(op): int(n) for op, n in calls.items()}
                for phase, calls in profiler.audit.calls.items()
            },
        }
        return total, detail
    finally:
        profiler.abort()
        model.train(was_training)
