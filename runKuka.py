#%%
methods=[
    {'name':'Naive', 'module':'other.train_naive'},
    {'name':'STGAT-MAD', 'module':'reconstruction.train_stgat_mad'},
    {'name':'Donut', 'module':'generative.vae.train_donut'},
    {'name':'LSTM-P', 'module':'prediction.train_lstm_prediction_malhotra'},
    {'name':'GDN', 'module':'prediction.train_gdn'},
    {'name':'TCN-S2S-P', 'module':'prediction.train_tcn_prediction_he'},
    {'name':'DeepAnt', 'module':'prediction.train_tcn_prediction_munir'},
    {'name':'LSTM-AE-OC-SVM', 'module':'other.train_lstm_ae_ocsvm'},
    {'name':'MTAD-GAT', 'module':'other.train_mtad_gat'},
    {'name':'THOC', 'module':'other.train_thoc'},
    {'name':'LSTM-AE', 'module':'reconstruction.train_lstm_ae'},
    {'name':'GenAD*', 'module':'reconstruction.train_genad'},
    {'name':'kNN', 'module':'baselines.train_knn'},
    {'name':'Varade', 'module':'prediction.train_varade'},
    {'name':'Isolation Forest', 'module':'baselines.train_iforest'},
    {'name':'GBRF', 'module':'prediction.train_gbrf'},
    {'name':'MSCRED*', 'module':'reconstruction.train_mscred'},
]

import sys
method=methods[int(sys.argv[1])]
print(f'### {method["name"]} - Start')
#%%
import importlib
import time
module=method['module']
packageName=f"timesead_experiments.{module}"
package = importlib.import_module(packageName)
experiment = package.experiment
get_test_pipeline=package.get_test_pipeline
get_batch_dim=package.get_batch_dim
get_anomaly_detector=package.get_anomaly_detector

from timesead.data.transforms import make_pipe_from_dict
from timesead.data.kuka_dataset import KukaDataset as Dataset
from timesead.data.transforms.dataset_source import DatasetSource
import torch

def main():
    #%%
    config_updates={
        'dataset':{
            'name': 'KukaDataset',
        },
        'training':{
            'device':'cpu',
        },
    }

    res=experiment.run(config_updates=config_updates)
    #%%
    if config_updates['dataset']['name']=='SMDDataset':
        sweep='server_id'
        values=[config_updates['dataset']['ds_args']['server_id']]
        additionalArgs={'training':False}
    elif config_updates['dataset']['name']=='ExathlonDataset':
        sweep='app_id'
        values=[config_updates['dataset']['ds_args']['app_id']]
        additionalArgs={'training':False}
    elif config_updates['dataset']['name']=='KukaDataset':
        sweep='otherSet'
        values=['collision','weight','velocity']
        additionalArgs={'training':False}

    prcs=[]
    for value in values:
        from timesead.evaluation import evaluator
        from timesead.data.dataset import collate_fn
        batchDimension=get_batch_dim()
        evalObj=evaluator.Evaluator()
        detector=res.result.object['detector'].object
        kwargs={}
        kwargs[sweep]=value
        kwargs.update(additionalArgs)
        val_loader=Dataset(**kwargs)
        testPipeline=get_test_pipeline()
        source=DatasetSource(val_loader)
        pipe=make_pipe_from_dict(testPipeline, source)
        pipeLoader=torch.utils.data.DataLoader(pipe,batch_size=4,collate_fn=collate_fn(batchDimension),drop_last=True)
        detector.eval()
        labels, scores = detector.get_labels_and_scores(pipeLoader)

        try:
            f1=evalObj.best_f1_score(labels,scores)
            auprc=evalObj.auprc(labels,scores)
            auroc=evalObj.auc(labels,scores)
            print(f'### {method["name"]} - {value} - AUPRC: {auprc}, F1: {f1}, AUROC: {auroc}')
            prcs.append(auroc[0])
        except:
            return 0.0

    prcs=torch.tensor(prcs)

    model=detector=res.result.object['model']
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"checkpoints/model_{timestamp}.pt"
    torch.save(model.state_dict(),filename)
    print(f'### {method["name"]} Mean - {(prcs[0]+prcs[1]+prcs[2])/3.0}')
# %%
main()