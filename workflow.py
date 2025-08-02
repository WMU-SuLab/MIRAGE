import torch
import tqdm
import random
from torch.cuda.amp import autocast
from utils.compute.metrics import count_metrics_binary_classification
from utils.records import test_metrics_record
from utils.task import binary_classification_task


def shuffle_genes_across_samples(inputs, gene_sizes, device, i):
    torch.manual_seed(i)
    num_samples, num_features = inputs.shape
    shuffled_inputs = inputs.clone()
    start_idx = 0

    for gene_size in gene_sizes:
        gene_snps = shuffled_inputs[:, start_idx:start_idx + gene_size].clone()
        for j in range(num_samples):
            random.shuffle(gene_snps[j].tolist())
        shuffled_inputs[:, start_idx:start_idx + gene_size] = gene_snps
        start_idx += gene_size

    return torch.tensor(shuffled_inputs, device=device) 

def train_valid_workflow(device, net, criterion, optimizer,  data_loader_iter, data_loaders, phase,model_name,gene_size,i,scaler=None,lasso=False):
    if phase == 'train':
        net.train()
    else:
        net.eval()
    y_true, y_pred, y_score = [], [], []
    running_loss = 0.0
    for inputs, labels in data_loader_iter:
        inputs = [each_input for each_input in inputs]
        #print('inputs',inputs)
        shuffled_inputs = shuffle_genes_across_samples(inputs[0], gene_size, device,i)
        #print('shuffled_inputs',shuffled_inputs)
        labels = labels.to(device)
        y_true += labels.int().reshape(-1).tolist()
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            with autocast(): 
                    outputs = net(shuffled_inputs)
                    loss, y_pred_batch, y_score_batch = binary_classification_task(outputs, labels, criterion=criterion)
            if phase == 'train':
                if not scaler:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        y_pred += y_pred_batch
        y_score += y_score_batch
        running_loss += loss.item()
    epoch_loss = running_loss / len(data_loaders[phase].dataset)
    all_metrics = count_metrics_binary_classification(y_true, y_pred, y_score)
    return epoch_loss, all_metrics

def test_workflow(device, net, data_loaders,model_name, writer,gene_size,i):
    net.eval()
    y_true, y_pred, y_score = [], [], []
    i=1
    for inputs, labels in tqdm.tqdm(data_loaders):
        inputs = [each_input.to(device) for each_input in inputs]
        shuffled_inputs = shuffle_genes_across_samples(inputs[0], gene_size, device,i)
        #inputs = [each_input.reshape(-1, 1, 179712) for each_input in inputs]
        labels = labels.to(device)
        y_true += labels.int().reshape(-1).tolist()
        with torch.no_grad():
            outputs = net(shuffled_inputs)
            y_pred_batch, y_score_batch = binary_classification_task(outputs, labels)
            y_pred += y_pred_batch
            y_score += y_score_batch
    test_metrics_record(y_true, y_pred, y_score, writer)

workflows = {
    'train': train_valid_workflow,
    'valid': train_valid_workflow,
    'test': test_workflow,
}
__all__ = ['workflows']