import copy
import os
import argparse
import time
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler
from base_code.base import checkpoints_dir, logs_dir
from base_code.divide_dataset import mk_dataset_paths
from init import init_net, init_strategy
from utils import setup_seed
from utils.dir import mk_dir
from utils.mk_data_loaders import mk_data_loaders_single_funcs
from utils.records import experiments_record, train_epoch_record
from utils.time import datetime_now_str
from utils.workflow import workflows
def main(
        train_dir_prefix: str, model_name: str, dataset_dir_path: str, snp_number: int,
        pretrain_checkpoint_path=None, pretrain_image_feature_checkpoint_path=None,
        dataset_in_memory=True, persistent_workers=True,
        gene_freq_file_path=None, label_data_id_field_name='participant_id', label_data_label_field_name='label',
        seed=2024, lr=1e-4, step_size=18,gamma=0.2, last_epoch=-1,
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
    net = init_net(device, model_name, snp_number, pretrain_checkpoint_path, pretrain_image_feature_checkpoint_path)
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
    data_loaders_func = mk_data_loaders_single_funcs[model_name]
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
            print(phase),
            epoch_loss, all_metrics = workflows[phase](
                device, net, criterion, optimizer, tqdm.tqdm(data_loaders[phase]), data_loaders, phase,model_name,scaler=scaler,lasso=False
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
        workflows['test'](device, net, data_loaders['test'], model_name,writer)
    else:
        workflows['test'](device, net, data_loaders['valid'],model_name,writer)


parser = argparse.ArgumentParser(description='MIRAGE train pipline')
parser.add_argument('--train_dir_prefix', required=False,default=datetime_now_str(), help='Output folder for results')
parser.add_argument('--model_name', required=False,default='SNPImageNet', help='Model selection among SNPImageNet, SNPNet and ImageNet (default: SNPImageNet)')
parser.add_argument('--dataset_dir_path', required=True, help='Input folder of datasets')
parser.add_argument('--snp_number', required=False,default=None,type=int,help='SNP number for training (default: None)')
parser.add_argument('--batch_size',required=False,default=32,type=int, help='Batch size (default:32)')
parser.add_argument('--label_data_id_field_name', required=False,default='participant_id' ,help='Colname of sample ID (default: participant_id)')
parser.add_argument('--label_data_label_field_name', required=False, default='high_myopia',help='Colname of sample label (default: label)')


args = parser.parse_args()

if __name__ == '__main__':
    train_on_gpu = torch.cuda.is_available()
    print("GPU is available：", train_on_gpu)
    if train_on_gpu:
        print('CUDA is available!')
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 
        # tensor.to("cuda") = tensor.cuda()
        gpu_device = torch.device("cuda")
        # gpu_device = torch.device("cuda:0")
        gpu_count = torch.cuda.device_count()
        print("GPU numbers:", gpu_count)
        print("CUDA version:", torch.version.cuda)
        # print("GPU index：", torch.cuda.current_device())
        # torch.cuda.set_device(gpu_default_device_number)
    else:
        print('CUDA is not available. Training on CPU ...')
        cpu_device = torch.device("cpu")
    main( train_dir_prefix=args.train_dir_prefix,
      model_name=args.model_name,
      dataset_dir_path=args.dataset_dir_path,#r"C:\Users\76754\Downloads\Multimodal",#"/pub/data/gss/MIRAGE/code/SuLabCohort/20240918215106/",#"/pub/data/gaoss/New_Multi/code/SuLabCohort/20240914111247/",
      snp_number= args.snp_number,        # college 100760    179712
      batch_size=args.batch_size,
      label_data_id_field_name=args.label_data_id_field_name,#学籍号  participant_id
      label_data_label_field_name=args.label_data_label_field_name)#图像和多模态high_myopia，遗传label


