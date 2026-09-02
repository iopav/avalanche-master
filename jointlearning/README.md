# Joint references and Intransigence

For each formal CIL run, create the matching cumulative Joint reference first. The matching identity is the literal dataset, order, seed, paired method, backbone ID, task groups, input view and training-parameter object; no hash is used.

```powershell
D:\anaconda3\envs\py310\python.exe -m jointlearning.run --dataset uwave --paired-method er_ace --order-id 1 --seed 62 --device cuda
D:\anaconda3\envs\py310\python.exe run_uwave.py --methods er_ace --order-ids 1 --seeds 62 --device cuda
```

The CIL runner requires exactly one matching Joint artifact and fills `cil_performance.intransigence` after training. `--skip-intransigence` permits a diagnostic run with `intransigence=null`; aggregation rejects it. Migrated ResNet18 hyperparameters must be revalidated and locked in `cil_experiments/final_hyperparameters.py` before either command is a formal run. TagFex now uses its source-structure image network and must be revalidated as well. Joint and its paired CIL run must select the same one of `resnet18_cifar_small`, `resnet18_cifar`, and `resnet18_cifar_large`.

After all requested order/seed combinations exist, generate the strict aggregate:

```powershell
D:\anaconda3\envs\py310\python.exe aggregate_formal_results.py --result-root result --output-prefix result/formal_aggregate
```
