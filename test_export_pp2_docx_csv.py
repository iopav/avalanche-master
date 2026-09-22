"""Run with python -m unittest test_export_pp2_docx_csv -v."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import export_pp2_docx_csv as report


class ExportTests(unittest.TestCase):
    def config(self):
        return dict(backbone=dict(backbone_id='resnet18_cifar',feature_dim=512,
                                 base_width=64,stage_channels=[64,128,256,512]),
                    method_parameters=dict(proj_hidden_dim=2048,proj_output_dim=1024))

    def test_accuracy_weights_and_final_row_not_diagonal(self):
        payload=dict(tasks=2,accuracy_matrix_lower_triangular=[[1,None],[0,1]])
        metrics,curve=report.performance(payload,[1,3])
        self.assertEqual(curve,[1,.75])
        self.assertEqual(metrics['average_incremental_accuracy'],.875)
        self.assertEqual(metrics['final_average_accuracy'],.75)
        self.assertEqual(metrics['average_forgetting'],1)
        self.assertEqual(metrics['backward_transfer'],-1)
        self.assertNotEqual(curve[-1],sum(payload['accuracy_matrix_lower_triangular'][-1])/2)

    def test_dynamic_parameter_storage_known_architecture(self):
        config=self.config()
        self.assertEqual(report.backbone_parameters(config),11168832)
        groups=[[0,1,2],[3,4,5],[6,7,8],[9,10,11]]
        ordinary=report.parameter_bytes(config,groups,'ewc')
        self.assertEqual(ordinary,[44681484,44687640,44693796,44699952])
        tagfex=report.parameter_bytes(config,groups,'tagfex')
        self.assertEqual(tagfex[-1],242386252)
        self.assertEqual(len(set(tagfex)),4)
        self.assertGreater(tagfex[1],tagfex[0])
        config['backbone']['backbone_id']='unverified'
        with self.assertRaisesRegex(ValueError,'Unverified'):
            report.parameter_bytes(config,groups,'ewc')

    def test_csv_fields_local_flops_and_distinct_joint_ids(self):
        rows=report.task_rows('tagfex','tagfex',.1,2,52,[.9,.8],[100,200],[10,30])
        self.assertEqual(tuple(rows[0]),report.FIELDS)
        self.assertEqual(rows[1]['pertaskflops'],30)
        self.assertEqual(rows[1]['model_parameter_storage_bytes_corrected'],200)
        self.assertEqual(rows[1]['param_memory_kib'],200/1024)
        self.assertEqual(rows[0]['run_id'],'tagfex_lr01_o2_s52')
        first=report.task_rows('joint','ewc',.1,2,52,[.9],[100],[''])[0]
        second=report.task_rows('joint','icarl',.1,2,52,[.9],[100],[''])[0]
        self.assertEqual(first['method'],'joint')
        self.assertEqual(first['pertaskflops'],'')
        self.assertNotEqual(first['run_id'],second['run_id'])

    def test_unknown_weights_and_bad_flops_rejected(self):
        payload=dict(tasks=2,accuracy_matrix_lower_triangular=[[1,None],[0,1]])
        with self.assertRaises(ValueError):report.performance(payload,[0,3])
        with self.assertRaises(ValueError):
            report.validate_ops(dict(core_training_flops=3,learning_auxiliary_flops=4,overall_learning_flops=8))

    def test_synthetic_search_failure_and_joint_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);method='ewc';exp='test';dataset='texture';order=1;seed=52
            groups=((0,1,2),(3,));lrs=(.1,.05,.01)
            directory=root/'search_result_test/ewc/order1'
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')
            expected=dict(exp_name=exp,dataset=dataset,method=method,order=order,seed=seed)
            candidates={'.1':{}}
            candidates={'0.1':dict(status='failed',error_message='failure'),
                        '0.05':dict(status='completed',overall_learning_flops=100,selection_loss=.1),
                        '0.01':dict(status='completed',overall_learning_flops=200,selection_loss=.2)}
            search=dict(expected,status='completed',lr_candidates=list(lrs),best_lr=.05,
                        candidates=candidates,total_search_flops=None,search_flops_lower_bound=300)
            write(directory/'order1_seed052_search.json',search)
            matrix=dict(tasks=2,accuracy_matrix_lower_triangular=[[1,None],[0,1]],test_samples_per_task=[1,3],
                        dataset=dataset,method=method,exp_name=exp,order_id=order,seed=seed)
            metrics,curve=report.performance(matrix,[1,3]);metrics['task_end_seen_accuracy_curve']=curve
            config=dict(self.config(),exp_name=exp,method=method,dataset=dict(name=dataset),
                        selected_task_groups=[list(g) for g in groups])
            size=report.parameter_bytes(config,groups,method)[-1]
            storage=dict(model_parameter_bytes=size,auxiliary_bytes=10,replay_sample_bytes=0,replay_label_bytes=0,total_bytes=size+10)
            for lr in (.05,.01):
                token=str(lr).replace('.','');stem=f'search_texture_ewc__order-1__lr-{token}__seed-052'
                total=candidates[str(lr)]['overall_learning_flops']
                config['final_hyperparameters']=dict(resolved=dict(learning_rate=lr))
                write(directory/f'lr{token}/{stem}__summary.json',dict(exp_name=exp,seed=seed,tasks=2,
                      config=config,cil_performance=metrics,selection=dict(loss=candidates[str(lr)]['selection_loss']),
                      training_operations=dict(core_training_flops=total-10,learning_auxiliary_flops=10,overall_learning_flops=total),persistent_storage=storage))
                write(directory/f'lr{token}/{stem}__accuracy-matrix.json',matrix)
            write(root/'joint_result_test/ewc/order1/joint_texture_ewc__order-1__seed-052.json',
                  dict(expected,status='completed',best_lr=.05,tasks=2,task_groups=[list(g) for g in groups],
                       accuracy_matrix_lower_triangular=[[.8,None],[.4,.9]],
                       training_operations=dict(core_training_flops=90,learning_auxiliary_flops=10,overall_learning_flops=100),
                       persistent_storage=dict(model_parameter_bytes=size,model_buffer_bytes=6,optimizer_state_bytes=4,total_bytes=size+10)))
            data=report.load_dataset(root,dataset,exp,{1:groups},[seed],lrs,['ewc'])
            self.assertEqual(len(data['csv_rows']),6)
            self.assertEqual(len(data['failures']),1)
            self.assertEqual(data['selected'][0]['search_extra'],200)
            self.assertEqual(data['selected'][0]['total'],300)
            self.assertFalse(data['selected'][0]['complete'])
            self.assertAlmostEqual(data['selected'][0]['metrics']['intransigence_mean'],-.15)
            self.assertTrue(all(abs(r['metrics']['intransigence_mean']+.15)<1e-12
                                for r in data['runs'] if r['status']=='completed'))
            self.assertTrue(all(r['pertaskflops']=='' for r in data['csv_rows']))
            self.assertEqual(report.make_sections(data)[0][2][0][2],'FAIL')
            sections=report.make_sections(data)
            self.assertIn('Total train\nFLOPs\nE12',sections[0][1])
            self.assertIn('Total learn\nFLOPs\nE12',sections[3][1])
            self.assertEqual(sections[4][2][0][sections[4][1].index('Intrans.')],'—')
            self.assertTrue(all(len(row)==len(headers) for _,headers,rows in sections for row in rows))
            search['search_flops_lower_bound']=999
            write(directory/'order1_seed052_search.json',search)
            with self.assertRaisesRegex(ValueError,'lower bound mismatch'):
                report.load_dataset(root,dataset,exp,{1:groups},[seed],lrs,['ewc'])


if __name__=='__main__':unittest.main()
