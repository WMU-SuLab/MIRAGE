import click
import torch
import os
import sys
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from workflow import workflows
sys.path.append("/pub/data/gaoss/New_Multi/code")
from utils import setup_seed
from utils.time import datetime_now_str
from base_code.divide_dataset import mk_dataset_paths
from init import init_net
from base_code.base import load_test
from utils.mk_data_loaders import mk_data_loaders_single_funcs
def main(test_dir_prefix,model_name,dataset_dir_path,wts_path,snp_number,contribution_path,dataset_in_memory=True, persistent_workers=True,
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
    data_loaders = data_loaders_func(**data_loaders_func_kwargs)  # 创建数据加载器
    load_test_dir = os.path.join(load_test, test_dir_prefix)
    writer = SummaryWriter(log_dir=load_test_dir)
    #best_model_wts=torch.load(best_model_checkpoints_path)['model']
    #net.load_state_dict(best_model_wts)
    net.eval()
    contrib_snp_list = []
    contrib_image_list = []
    y_pred,snp_pred,image_pred,prob_snp_image,prob_snp,prob_image,y_true = workflows['valid'](device, net, data_loaders['valid'])
    workflows['test'](device, net,data_loaders['test'],writer)
    for true,pred, snp, image in zip(y_true, y_pred,snp_pred, image_pred):
        snp_image =0.0
        drop_snp = 0.0
        drop_image = 0.0
        if true == pred:
            snp_image = 2
        if true == snp:
            drop_image = 1
        if true == image:
            drop_snp = 1 
        contrib_snp = (drop_image+snp_image-drop_snp)/2.0
        contrib_image = (drop_snp+snp_image-drop_image)/2.0   
        contrib_snp_list.append(contrib_snp)
        contrib_image_list.append(contrib_image)
    df = pd.DataFrame({
        'contrib_snp': contrib_snp_list,
        'contrib_image': contrib_image_list
    }) 
    print(df)
    df.to_excel(os.path.join(contribution_path,'contribution.xlsx'), index=False)


if __name__ == '__main__':
    main( test_dir_prefix=datetime_now_str(),
          model_name='SNPImageNet',
          dataset_dir_path=r"/pub/data/gaoss/New_Multi/code/SuLabCohort/20240918215106/",
          wts_path=r"/pub/data/gaoss/New_Multi/code/work_dirs/records/checkpoints/20241219214429/best_model_checkpoints.pth" ,
          snp_number=100760,   #75988,#295552 #296180 230608 179712
          contribution_path=r"/pub/data/gaoss/New_Multi/code/model_contribution_code/",
          batch_size=32,
          label_data_id_field_name="学籍号", #学籍号  participant_id
          label_data_label_field_name='high_myopia')
          



