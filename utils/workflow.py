import torch
import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from .compute.metrics import count_metrics_binary_classification
from .multi_gpus import barrier, reduce_value
from .records import test_metrics_record
from .task import binary_classification_task, multi_classification_task


def custom_loss(snp, image, snp_image, Y, We, Wg, Wm, a, b):
    snp = torch.sigmoid(snp)
    image = torch.sigmoid(image)
    snp_image = torch.sigmoid(snp_image)
    fusion_output_error = F.mse_loss(snp_image, Y)
    #single_output_error = F.mse_loss(snp + image, Y)
    snp_error = F.mse_loss(snp,Y)
    image_error = F.mse_loss(image,Y)
    l1_reg_We = a * torch.norm(We, p=1)
    l1_reg_Wg = a * torch.norm(Wg, p=1)
    consistency_loss = b * F.mse_loss(snp, image)
    total_loss = (fusion_output_error +
                  snp_error  +      #snp_error +  image_error +               
                  image_error +
                  l1_reg_We + 
                  l1_reg_Wg + 
                  consistency_loss) 
    return total_loss

def train_valid_workflow(device, net, criterion, optimizer,  data_loader_iter, data_loaders, phase,model_name,scaler=None,lasso=False,
                         multi_gpu: bool = False):
    if phase == 'train':
        net.train()
    else:
        net.eval()
    y_true, y_pred, y_score = [], [], []
    running_loss = 0.0
    for inputs, labels in data_loader_iter:
        inputs = [each_input.to(device) for each_input in inputs]
        #inputs = [each_input.reshape(-1, 1, 179712) for each_input in inputs]
        labels = labels.to(device)
        #permuted_indices = torch.randperm(labels.size(0))
        #labels = labels[permuted_indices]
        y_true += labels.int().reshape(-1).tolist()
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            with autocast():
                if model_name=='SNPImageNet':
                      snp,image,snp_image = net(*inputs)
                      _, y_pred_batch, y_score_batch = binary_classification_task(snp_image, labels, criterion=criterion)
                      Ws = torch.cat([p.view(-1) for p in net.multi_modal_model.SNP.parameters()])
                      Wi = torch.cat([p.view(-1) for p in net.multi_modal_model.IMAGE.parameters()])
                      Wm = torch.cat([p.view(-1) for p in net.multi_modal_model.encoder2.parameters()])
                      loss = custom_loss(snp, image, snp_image, labels, Ws, Wi, Wm,a=0.03,b=0.5) 
                      #print("##############loss##############",loss)
                else:  
                    outputs = net(*inputs)
                    if lasso ==True:
                        y_pred_batch, y_score_batch = binary_classification_task(outputs,labels,criterion=None)
                        loss=net.snp_model.lasso_loss(outputs, labels)
                    else:
                        loss, y_pred_batch, y_score_batch = binary_classification_task(outputs, labels, criterion=criterion)
            # loss, y_pred_batch, y_score_batch = multi_classification_task(outputs, labels, criterion=criterion)
            if phase == 'train':
                if not scaler:
                    loss.backward()
                    optimizer.step()
                else:
                    # Scales loss
                    scaler.scale(loss).backward()
                    # scaler.step()
                    scaler.step(optimizer)
                    scaler.update()
        if multi_gpu:
            barrier()
            loss = reduce_value(loss)
        y_pred += y_pred_batch
        y_score += y_score_batch
        running_loss += loss.item()
    epoch_loss = running_loss / len(data_loaders[phase].dataset)
    all_metrics = count_metrics_binary_classification(y_true, y_pred, y_score)
    return epoch_loss, all_metrics


def test_workflow(device, net, data_loaders,model_name, writer):
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
            if model_name=='SNPImageNet':
                net.multi_modal_model.is_test_phase = True
                _,_,outputs = net(*inputs)#snp,image,
                y_pred_batch, y_score_batch = binary_classification_task(outputs, labels)
                #gate_values1.append(net.multi_modal_model.z1_gate.detach().cpu().numpy())
                #gate_values2.append(net.multi_modal_model.z2_gate.detach().cpu().numpy())
                #feature_contributions.append(net.multi_modal_model.out.detach().cpu().numpy())
            else:
                outputs = net(*inputs)
                y_pred_batch, y_score_batch = binary_classification_task(outputs, labels)
            # y_pred_batch, y_score_batch = multi_classification_task(outputs, labels)
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
