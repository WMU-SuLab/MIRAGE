import os
import csv
import torch
import copy
import time
import sys
import tqdm
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler
sys.path.append("/pub/data/gaoss/New_Multi/code")
from base_code.base import checkpoints_dir, logs_dir
from base_code.divide_dataset import mk_dataset_paths
from models import nets
from init import  init_strategy
from utils import setup_seed
from utils.dir import mk_dir
from utils.mk_data_loaders import mk_data_loaders_single_funcs
from utils.records import experiments_record, train_epoch_record
from utils.time import datetime_now_str
from utils.workflow import workflows
from plink.genome_local_net import DeepcvGRS
from models.InteractionModel import Gene_Block
from interactionfunctions import  GlobalSIS


def Init_net(device, model_name, gene_block, deepcvGRS,pretrain_checkpoint_path, pretrain_image_feature_checkpoint_path: str = None):
    print(model_name)
    if Net := nets.get(model_name, None):
        net = Net(gene_block,deepcvGRS)
        print(net)
    else:
        raise Exception(f"model_name {model_name} not found")
    net.to(device)
    if pretrain_checkpoint_path and os.path.exists(pretrain_checkpoint_path) and os.path.isfile(pretrain_checkpoint_path):
        # missing_keys, unexpected_keys = net.load_state_dict(
        #     torch.load(pretrain_checkpoint_path, map_location=device)['state_dict'], strict=True)
        missing_keys, unexpected_keys = net.load_state_dict(
            torch.load(pretrain_checkpoint_path, map_location=device)['model'], strict=True)
    if pretrain_image_feature_checkpoint_path and os.path.exists(pretrain_image_feature_checkpoint_path) \
            and os.path.isfile(pretrain_image_feature_checkpoint_path):
        missing_keys, unexpected_keys = net.load_image_feature_state_dict(
            torch.load(pretrain_image_feature_checkpoint_path, map_location=device))
    return net


def main(
        train_dir_prefix: str, model_name: str, dataset_dir_path: str, snps_size_dir:str,output_dir:str,snp_number: int,
        pretrain_checkpoint_path=None, pretrain_image_feature_checkpoint_path=None,
        dataset_in_memory=True, persistent_workers=True,
        gene_freq_file_path=None, label_data_id_field_name='participant_id', label_data_label_field_name='label',
        seed=2024, lr=1e-3, step_size=18,gamma=0.2, last_epoch=-1,
        epochs=200, batch_size=32,  log_interval=1,save_interval=10,
        use_early_stopping=True, early_stopping_step=7, early_stopping_delta=0,    
        cuda_device=0, remarks=None
):
          
    if cuda_device:
        torch.cuda.set_device(cuda_device)
        device = torch.device(f"cuda:{cuda_device}")
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu")
    print(f'current device:{device}')
    setup_seed(seed)
    ## Load the gene structure
    with open(snps_size_dir, newline='') as f:
        reader = csv.reader(f)
        tmp = list(reader)
    gene_size = [int(float(i[0])) for i in tmp]
    print("gene_size)",gene_size)
    num_gene = len(gene_size)
    print("num_gene",num_gene)
    gene_block = Gene_Block(gene_size, device = device)
    deepcvGRS = DeepcvGRS(num_gene)
    net = Init_net(device, model_name,gene_block,deepcvGRS, pretrain_checkpoint_path, pretrain_image_feature_checkpoint_path)
    optimizer, scheduler, criterion, loss_early_stopping = init_strategy(
        net, lr, step_size, gamma, last_epoch, pretrain_checkpoint_path, early_stopping_step, early_stopping_delta)
    log_dir_path = os.path.join(logs_dir, train_dir_prefix)
    mk_dir(log_dir_path)
    experiments_record(
        os.path.join(log_dir_path, 'experiments_records.txt'),
        train_dir_prefix, model_name, dataset_dir_path, snp_number,
        epochs, batch_size, lr, step_size, criterion, gamma, save_interval, log_interval, remarks)
    model_checkpoints_dir = os.path.join(checkpoints_dir, train_dir_prefix)
    print(f'model_checkpoints',model_checkpoints_dir)
    mk_dir(model_checkpoints_dir)
    best_model_checkpoints_path = os.path.join(model_checkpoints_dir, 'best_model_checkpoints.pth')
    print(best_model_checkpoints_path)
    writer = SummaryWriter(log_dir=log_dir_path)
    # writer.add_graph(net, nets_fake_data(device, model_name, batch_size, snp_number))
    data_paths = mk_dataset_paths(dataset_dir_path)
    data_loaders_func = mk_data_loaders_single_funcs["SNPNet"]
    data_loaders_func_kwargs = {'data_paths': data_paths, 'batch_size': batch_size}
    if dataset_in_memory:
        data_loaders_func_kwargs['in_memory'] = dataset_in_memory
    if persistent_workers:
        data_loaders_func_kwargs['persistent_workers'] = persistent_workers
    if gene_freq_file_path:
        data_loaders_func_kwargs['gene_freq_file_path'] = gene_freq_file_path
    if label_data_id_field_name:
        data_loaders_func_kwargs['label_data_id_field_name'] = label_data_id_field_name
    if label_data_label_field_name:
        data_loaders_func_kwargs['label_data_label_field_name'] = label_data_label_field_name
    data_loaders = data_loaders_func(**data_loaders_func_kwargs)
    scaler = GradScaler()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_f1 = 0
    since = time.time()
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            epoch_loss, all_metrics = workflows[phase](
                device, net, criterion, optimizer, tqdm.tqdm(data_loaders[phase]), data_loaders, phase,scaler=scaler,lasso=False
            )
            f1, best_f1, best_model_wts = train_epoch_record(
                epoch_loss, all_metrics, net, optimizer, epoch, epochs, phase,
                writer, log_interval, best_f1, best_model_wts, best_model_checkpoints_path, since)
            if use_early_stopping and phase == 'valid':
                loss_early_stopping(epoch_loss)
        scheduler.step()
        if epoch % step_size == 0:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        if epoch % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model': net.state_dict(),
                'f1': f1,
                'optimizer': optimizer.state_dict()
            }, os.path.join(model_checkpoints_dir, f'epoch_{epoch}_model_checkpoints.pth'))
        if use_early_stopping and loss_early_stopping.early_stop:
            break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val f1: {:4f}'.format(best_f1))
    net.load_state_dict(best_model_wts)

    if data_loaders.get('test', None):
        workflows['test'](device, net, data_loaders['test'], writer)
    else:
        workflows['test'](device, net, data_loaders['valid'],writer)
    

if __name__ == '__main__':
    train_on_gpu = torch.cuda.is_available()
    print("Is the GPU available：", train_on_gpu)
    if train_on_gpu:
        print('CUDA is available!')
        # GPU
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        gpu_device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        print("GPU number:", gpu_count)
        print("CUDA version:", torch.version.cuda)
    else:
        print('CUDA is not available. Training on CPU ...')
        # CPU
        # tensor_gpu.to("cpu") = tensor_gpu.cpu()
        cpu_device = torch.device("cpu")
    main( train_dir_prefix=datetime_now_str(),
      model_name='InteractionNet',
      dataset_dir_path=r"/pub/data/gaoss/New_Multi/code/MAGIC_prs_20/75_HM/",#"/pub/data/gaoss/New_Multi/code/shap_top_snp/common/data/",
      snps_size_dir=r"/pub/data/gaoss/New_Multi/code/MAGIC_prs_20/75_HM/HM_gene_size.txt",#"/pub/data/gaoss/New_Multi/code/shap_top_snp/common/snp_size.txt",
      output_dir=r"/pub/data/gaoss/New_Multi/code/Interaction_premutation/75_HM/",#"/pub/data/gaoss/New_Multi/code/Interaction_premutation/top1000/",
      snp_number=924,
      batch_size=32,
      label_data_id_field_name="participant_id",
      label_data_label_field_name='label')