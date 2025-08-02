import torch
import os
from torch.utils.tensorboard import SummaryWriter
from utils import setup_seed
from shap_code.shap_value import SNP_Shap_Value
from utils.time import datetime_now_str
from base_code.divide_dataset import mk_dataset_paths
from init import init_net
from base_code.base import load_test
from utils.mk_data_loaders import mk_data_loaders_single_funcs
from utils.workflow import workflows
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
        workflows['test'](device, net,data_loaders['test'],model_name,writer)
    else:
        workflows['test'](device, net, data_loaders['valid'],model_name,writer)
    if model_name=="SNPNet":
        SNP_Shap_Value(net,data_loaders['train'], data_loaders['valid'], device,test_dir_prefix,valid=True)


if __name__ == '__main__':
    main( test_dir_prefix=datetime_now_str(),
          model_name='SNPImageNet',
          dataset_dir_path=r"/pub/data/gaoss/New_Multi/code/SuLabCohort/20240918215106/",
          wts_path=r"/pub/data/gaoss/New_Multi/code/work_dirs/records/checkpoints/20241219214429/best_model_checkpoints.pth" ,
          snp_number=100760,   #75988,#295552 #296180 230608 179712
          batch_size=32,
          label_data_id_field_name="学籍号", #学籍号  participant_id
          label_data_label_field_name='high_myopia')#图像和多模态high_myopia，遗传label

