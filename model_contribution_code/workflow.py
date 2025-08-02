import torch
import tqdm
import sys
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
sys.path.append("/pub/data/gaoss/New_Multi/code")
from utils.compute.metrics import count_metrics_binary_classification
from utils.multi_gpus import barrier, reduce_value
from utils.records import test_metrics_record
from utils.task import binary_classification_task, multi_classification_task
 

def train_valid_workflow(device, net,data_loader_iter):
    net.eval()
    net.multi_modal_model.drop_snp = True
    net.multi_modal_model.drop_image = True
    net.multi_modal_model.calculate_contribution = True
    y_true, y_pred,snp_pred,image_pred, y_score,prob_snp_image,prob_snp,prob_image = [], [], [], [], [],[],[],[]
    for inputs, labels in data_loader_iter:
        inputs = [each_input.to(device) for each_input in inputs]
        labels = labels.to(device)
        y_true += labels.int().reshape(-1).tolist()   
        snp,image,snp_image,drop_image,drop_snp = net(*inputs)
        y_pred_batch, y_score_batch = binary_classification_task(snp_image, labels)
        snp_pred_batch, _ = binary_classification_task(drop_image, labels)
        image_pred_batch, _ = binary_classification_task(drop_snp, labels)
        
        prob_snp_image_batch = torch.sigmoid(snp_image).cpu().reshape(-1).tolist()
        prob_snp_batch = torch.sigmoid(drop_image).cpu().reshape(-1).tolist()
        prob_image_batch = torch.sigmoid(drop_snp).cpu().reshape(-1).tolist()
        
        y_pred += y_pred_batch
        snp_pred+=snp_pred_batch
        image_pred+=image_pred_batch
        prob_snp_image+=prob_snp_image_batch
        prob_snp+=prob_snp_batch
        prob_image+=prob_image_batch
        y_score += y_score_batch
    all_metrics = count_metrics_binary_classification(y_true, y_pred, y_score)
    print(f"Valid Metrics: \n"
        f"acc: {all_metrics['acc']:.4f} | precision: {all_metrics['precision']:.4f} | recall: {all_metrics['recall']:.4f} | \n"
        f"f1: {all_metrics['f1']:.4f} | mcc: {all_metrics['mcc']:.4f} | auc: {all_metrics['auc_score']:.4f}  | \n"
        f"tpr: {all_metrics['tpr']:.4f} | fpr: {all_metrics['fpr']:.4f} | ks: {all_metrics['ks']:.4f} | sp: {all_metrics['sp']:.4f}\n"
    )
    return y_pred,snp_pred,image_pred,prob_snp_image,prob_snp,prob_image,y_true


def test_workflow(device, net, data_loaders, writer):
    #gate_values1=[]
    #gate_values2=[]
    #feature_contributions=[]
    net.eval()
    y_true, y_pred, y_score = [], [], []
    i=1
    for inputs, labels in tqdm.tqdm(data_loaders):
        inputs = [each_input.to(device) for each_input in inputs]
        #inputs = [each_input.reshape(-1, 1, 179712) for each_input in inputs]
        labels = labels.to(device)
        #permuted_indices = torch.randperm(labels.size(0))
        #labels = labels[permuted_indices]
        y_true += labels.int().reshape(-1).tolist()
        with torch.no_grad():
            net.multi_modal_model.is_test_phase = True
            _,_,outputs,_,_ = net(*inputs)#snp,image,
            y_pred_batch, y_score_batch = binary_classification_task(outputs, labels)
            #gate_values1.append(net.multi_modal_model.z1_gate.detach().cpu().numpy())
            #gate_values2.append(net.multi_modal_model.z2_gate.detach().cpu().numpy())
            #feature_contributions.append(net.multi_modal_model.out.detach().cpu().numpy())
            y_pred += y_pred_batch
            y_score += y_score_batch
    test_metrics_record(y_true, y_pred, y_score, writer)
    #gate_snp = np.concatenate(gate_values1, axis=0)
    #gate_image = np.concatenate(gate_values2, axis=0)
    #print("gate_snp",gate_snp.shape)
    #print("gate_image",gate_image.shape)
    #np.save("/pub/data/gaoss/New_Multi/code/weight_explainability/gate_snp.npy", gate_snp) 
    #np.save("/pub/data/gaoss/New_Multi/code/weight_explainability/gate_image.npy", gate_image)
    #feature_contributions = np.concatenate(feature_contributions, axis=0)
    #print("feature_contributions",feature_contributions.shape)
    #encoder_weights = net.multi_modal_model.encoder2[0].weight.detach().cpu().numpy()
    #np.save("/pub/data/gaoss/New_Multi/code/weight_explainability/feature_contributions.npy", feature_contributions)
    #np.save("/pub/data/gaoss/New_Multi/code/weight_explainability/encoder_weights.npy", encoder_weights)


workflows = {
    'train': train_valid_workflow,
    'valid': train_valid_workflow,
    'test': test_workflow,
}

__all__ = ['workflows']