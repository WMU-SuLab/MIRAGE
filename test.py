import torch
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from utils import setup_seed
from shap_code.shap_value import SNP_Shap_Value
from utils.time import datetime_now_str
from base_code.divide_dataset import mk_dataset_paths
from init import init_net
from base_code.base import load_test
from utils.mk_data_loaders import mk_data_loaders_single_funcs
from utils.workflow import workflows

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
def main(test_dir_prefix,model_name,dataset_dir_path,wts_path,snp_number,dataset_in_memory=True, persistent_workers=True,
         gene_freq_file_path=None, label_data_id_field_name='participant_id', label_data_label_field_name='label',batch_size=32):
    print(f'time:{test_dir_prefix}')
    setup_seed(2023)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = init_net(device, model_name, snp_number, pretrain_checkpoint_path=wts_path)
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
    load_test_dir = os.path.join(load_test, test_dir_prefix)
    writer = SummaryWriter(log_dir=load_test_dir)
    #best_model_wts=torch.load(best_model_checkpoints_path)['model']
    #net.load_state_dict(best_model_wts)
    net.eval()
    if data_loaders.get('test', None):
        print("##############Start test################")
        workflows['test'](device, net,data_loaders['test'],model_name,writer)
    else:
        print("##############Start test 2################")
        workflows['test'](device, net, data_loaders['valid'],model_name,writer)
    if model_name=="SNPNet":
        print("进入计算SNPshap")
        SNP_Shap_Value(net,data_loaders['train'], data_loaders['valid'], device,test_dir_prefix,valid=True)


parser = argparse.ArgumentParser(description='MIRAGE test pipline')
parser.add_argument('--test_dir_prefix', required=False,default=datetime_now_str(), help='Output folder for results')
parser.add_argument('--model_name', required=False,default='SNPImageNet', help='Model selection among SNPImageNet, SNPNet and ImageNet (default: SNPImageNet)')
parser.add_argument('--dataset_dir_path', required=True, help='Input folder of datasets')
parser.add_argument('--wts_path', required=True,help='Trained Model path (See example in `MIRAGE\base_code\example\best_model_checkpoints.pth`)')
parser.add_argument('--snp_number', required=False,default=100760,type=int,help='SNP number for training (default: 100760)')
parser.add_argument('--batch_size',required=False,default=32,type=int, help='Batch size (default:32)')
parser.add_argument('--label_data_id_field_name', required=False,default='participant_id' ,help='Column name of sample ID (default: participant_id)')
parser.add_argument('--label_data_label_field_name', required=False, default='high_myopia',help='Column name of sample label (default: label)')

args = parser.parse_args()

if __name__ == '__main__':
    main( test_dir_prefix=args.test_dir_prefix,
          model_name=args.model_name,
          dataset_dir_path=args.dataset_dir_path,
          wts_path=args.wts_path,
          snp_number=args.snp_number,   #75988,#295552 #296180 230608 179712
          batch_size=args.batch_size,
          label_data_id_field_name=args.label_data_id_field_name, #学籍号  participant_id
          label_data_label_field_name=args.label_data_label_field_name)#图像和多模态high_myopia，遗传label


