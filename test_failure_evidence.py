import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from cil_experiments.texture_ewc_failure import FailureEvidenceArtifacts, evidence_complete
from generate_pp2_dataset_tables import failure_cost


class FailureEvidenceTests(unittest.TestCase):
    def test_sparse_phase_counts_and_invalid_counts(self):
        cases = [({'core_training': 100}, 100, True),
                 ({'learning_auxiliary': 100}, 100, True),
                 ({}, 0, True),
                 ({'core_training': 100}, 101, False),
                 ({'unknown': 100}, 100, False),
                 ({'core_training': True}, 1, False),
                 ({'core_training': -1}, 0, False)]
        for phases, total, valid in cases:
            with self.subTest(phases=phases, total=total), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = dict(exp_name='test', final_hyperparameters=dict(resolved=dict(learning_rate=.1)))
                evidence = FailureEvidenceArtifacts(root, 'texture', 'er_ace', 1, 52, config,
                    False, Path('order1/lr01'), False, 'candidate', True)
                with self.assertRaises(FloatingPointError):
                    with evidence as artifacts:
                        artifacts.logger.info('task=1 wall_s=1 flops=%s', json.dumps(
                            dict(total_flops=total, phase_flops=phases)))
                        raise FloatingPointError('Non-finite training loss test')
                identity = dict(exp_name='test', dataset='texture', method='er_ace', order=1, seed=52)
                if valid:
                    self.assertEqual(failure_cost(root/'er_ace/order1', identity, 4)[0], total)
                else:
                    with self.assertRaises(ValueError):
                        failure_cost(root/'er_ace/order1', identity, 4)

    def test_completed_erace_replays_missing_evidence_once(self):
        from cil_experiments import lr_search as search
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root/'selected.json'
            selected.write_text(json.dumps({'inference': {'final_latency_ms_per_sample': 1}}))
            payload = dict(exp_name='test', dataset='texture', method='er_ace', order=1, seed=52,
                status='completed', best_lr=.01, best_checkpoint=str(selected),
                best_summary_file=str(selected), best_accuracy_matrix_file=str(selected),
                candidates={'0.1': dict(status='failed', error_type='FloatingPointError',
                    error_message='Non-finite training loss original'),
                    '0.01': dict(status='completed')})
            args = dict(project_root=root, dataset_root=root, search_root=root,
                exp_name='test', dataset='texture', method='er_ace', order_id=1, seed=52,
                device=torch.device('cpu'), backbone='resnet18_cifar',
                loss_selection='last_epoch_train_mean', lr_candidates=(.1,.01))
            def fail(**kwargs):
                config = dict(exp_name='test', final_hyperparameters=dict(resolved=dict(learning_rate=.1)))
                evidence = FailureEvidenceArtifacts(root, 'texture', 'er_ace', 1, 52, config,
                    False, kwargs['run_subdir'], False, 'candidate', True)
                error = FloatingPointError('Non-finite training loss replay')
                with evidence as artifacts:
                    artifacts.snapshot = lambda: dict(task=5, epoch=1, loss='nan', completed_task_wall_s=[1]*4, accuracy_matrix=[])
                    for task in range(1,5):
                        artifacts.logger.info('task=%s wall_s=1 flops=%s', task, json.dumps(dict(total_flops=30, phase_flops=dict(core_training=25, learning_auxiliary=5))))
                    raise error
            with patch.object(search, '_load_unit', return_value=payload), \
                 patch.object(search, 'get_final_hyperparameters', return_value={}), \
                 patch.object(search, 'validate_search_unit'), patch.object(search, 'validate_summary'), \
                 patch.object(search, 'run_experiment', side_effect=fail) as training:
                output = search.run_search_unit(**args)
                search.run_search_unit(**args)
                self.assertEqual(training.call_count, 1)
            saved = json.loads(output.read_text())
            self.assertEqual(saved['best_lr'], .01)
            self.assertEqual(saved['candidates']['0.1']['error_message'], 'Non-finite training loss replay')
            self.assertEqual(saved['candidates']['0.01'], {'status':'completed'})
            self.assertEqual(set(saved['candidates']['0.1']), {'status','error_type','error_message','traceback','failure_manifest'})
            identity = dict(exp_name='test', dataset='texture', method='er_ace', order=1, seed=52)
            directory = root/'er_ace/order1'
            self.assertEqual(failure_cost(directory, identity, 6)[0], 120)
            # An unreferenced duplicate replay must not double the replacement.
            import shutil
            chosen = Path(saved['candidates']['0.1']['failure_manifest']).parent
            self.assertEqual(chosen.parent, root/'er_ace/order1/lr01/failed_runs')
            # Existing nested archives migrate without retraining or changing data.
            from normalize_failure_layout import normalize
            legacy = chosen.parent.parent/'failure_recovery/replay/failed_runs'/chosen.name
            legacy.parent.mkdir(parents=True)
            chosen.rename(legacy)
            saved['candidates']['0.1']['failure_manifest'] = str(legacy/'manifest.json')
            output.write_text(json.dumps(saved))
            before = {p.name: p.read_bytes() for p in legacy.iterdir()}
            self.assertEqual(normalize(root), (1, 0))
            self.assertTrue(legacy.exists())
            self.assertEqual(normalize(root, apply=True), (1, 0))
            expected = json.loads(json.dumps(saved))
            expected['candidates']['0.1']['failure_manifest'] = str((chosen/'manifest.json').resolve())
            self.assertEqual(json.loads(output.read_text()), expected)
            self.assertEqual({p.name: p.read_bytes() for p in chosen.iterdir()}, before)
            self.assertFalse((chosen.parent.parent/'failure_recovery').exists())
            self.assertEqual(normalize(root, apply=True), (0, 0))
            self.assertEqual(failure_cost(directory, identity, 6)[0], 120)
            shutil.copytree(chosen, chosen.with_name('duplicate'))
            self.assertEqual(failure_cost(directory, identity, 6)[0], 120)

    def test_all_methods_keep_original_task_end_format(self):
        for method in ('er_ace', 'ewc', 'icarl', 'tagfex', 'fecam', 'cwr_star'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                config = dict(exp_name='test', final_hyperparameters=dict(resolved=dict(learning_rate=.1)))
                evidence = FailureEvidenceArtifacts(Path(tmp), 'texture', method, 1, 52, config,
                    False, Path('order1/lr01'), False, 'candidate', True)
                done = dict(total_flops=100, phase_flops=dict(core_training=80, learning_auxiliary=20))
                error = FloatingPointError('Non-finite training loss test')
                with self.assertRaises(FloatingPointError):
                    with evidence as artifacts:
                        artifacts.snapshot = lambda: dict(task=2, epoch=1, loss='nan', completed_task_wall_s=[1], accuracy_matrix=[])
                        artifacts.logger.info('task=1 wall_s=1 flops=%s', json.dumps(done))
                        raise error
                self.assertFalse(evidence.summary_path.exists())
                identity = dict(exp_name='test', dataset='texture', method=method, order=1, seed=52, lr=.1)
                record = dict(status='failed', error_type='FloatingPointError', error_message=str(error), failure_manifest=error.failure_manifest)
                self.assertTrue(evidence_complete(record, identity))
                expected = {k:v for k,v in identity.items() if k != 'lr'}
                total, notes = failure_cost(Path(tmp)/method/'order1', expected, 4)
                self.assertEqual(total, 100)
                self.assertEqual(len(notes), 1)
                # Detect corrupted archive instead of silently accepting cost.
                Path(error.failure_manifest).with_name('run.log').write_text('corrupted')
                self.assertFalse(evidence_complete(record, identity))
                with self.assertRaises(ValueError):
                    failure_cost(Path(tmp)/method/'order1', expected, 4)


if __name__ == '__main__':
    unittest.main()
