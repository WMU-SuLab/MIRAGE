import os
import click
import torch
import numpy as np
import pandas as pd
from base import load_test
from PIL import Image
from divide_dataset import mk_dataset_paths
from utils.time import datetime_now_str
from torch.utils.tensorboard import SummaryWriter
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from init import init_net
from utils import setup_seed
from utils.workflow import workflows
from utils.interpretability import make_img_cam, imshow, make_image_saliency_maps,make_gene_saliency_maps, make_gene_image_saliency_maps, \
    img_saliency_maps_show
from utils.transforms import gene_image_transforms
from utils.mk_data_loaders import mk_data_loaders_single_funcs


def main(test_dir_prefix,model_name: str, wts_path: str,
         snp_numbers: int, dataset_dir_path: str, image_file_path: str,gene_file_path:str,
         output_dir_path: str,saliency_maps_name: str, baseline_method: str, cam_name: str,
         dataset_in_memory=True, persistent_workers=True,label_data_id_field_name=None, 
         label_data_label_field_name=None,batch_size=32):
    print(f'time:{test_dir_prefix}')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    use_cuda = True if torch.cuda.is_available() else False
    setup_seed(2024)
    net = init_net(device, model_name, snp_numbers, pretrain_checkpoint_path=wts_path)
    if image_file_path:
        image = Image.open(image_file_path)
        img_tensor = gene_image_transforms['test'](image)
        img_tensor = img_tensor.to(device)
        if cam_name!='None':
            target_layers = [net.image_features.stages[-1][-1].dwconv]
            targets = [BinaryClassifierOutputTarget(1)]
            # targets = None
            img, grayscale_cam, visualization = make_img_cam(
                net, cam_name, target_layers, targets, img_tensor=img_tensor, use_cuda=use_cuda)
            imshow([img, grayscale_cam, visualization], ["image", "cam", "image + cam"],
                   file_name=f'{cam_name}_{os.path.basename(image_file_path)}')
        if saliency_maps_name!='None':
            image_attributions = make_image_saliency_maps(net, saliency_maps_name, img_tensor)
            img_saliency_maps_show(image_attributions, img_tensor, image_file_path, saliency_maps_name)
    if gene_file_path:
         data_paths = mk_dataset_paths(gene_file_path)
         data_loaders_func = mk_data_loaders_single_funcs[model_name]
         data_loaders_func_kwargs = {'data_paths': data_paths, 'batch_size': batch_size}
         if dataset_in_memory:
             data_loaders_func_kwargs['in_memory'] = dataset_in_memory
         if persistent_workers:
             data_loaders_func_kwargs['persistent_workers'] = persistent_workers
         if label_data_id_field_name:
             data_loaders_func_kwargs['label_data_id_field_name'] = label_data_id_field_name
         if label_data_label_field_name:
             data_loaders_func_kwargs['label_data_label_field_name'] = label_data_label_field_name
         data_loaders = data_loaders_func(**data_loaders_func_kwargs)
         n = 0
         all_weights = []
         for phase in ['train', 'valid']:
             if phase == 'valid':
                 for inputs, labels in data_loaders[phase]:
                     inputs = [each_input.to(device) for each_input in inputs]
                     gene_tensors= inputs[0]
                     for i in range(gene_tensors.shape[0]):
                         if n==0:
                             total_weight_sum=np.zeros(gene_tensors.shape[1])
                         gene_tensor = gene_tensors[i].unsqueeze(0)
                         print(gene_tensor.shape)
                         attributions = make_gene_saliency_maps(net, device,gene_tensor=gene_tensor,saliency_maps_name=saliency_maps_name,baseline_method=baseline_method)
                         gene_attribution= attributions[0]
                         print("gene_attribution",gene_attribution)
                         print("gene_attribution shape:", gene_attribution.shape)
                         weight = np.abs(gene_attribution.flatten().cpu().numpy())
                         print("weight",weight)
                         print("weight shape:", weight.shape)
                         total_weight_sum +=weight
                         n+=1
         print("sampel numpy",n)
         average_weight = total_weight_sum / n
         X_reshape = average_weight.reshape(1, average_weight.shape[0] // 4, 4)
         X_window_sum = np.sum(X_reshape, axis=2)
         X_window_sum = X_window_sum / 4
         X_flat = X_window_sum.flatten()
         print("mean result：",X_flat)
         df = pd.DataFrame()
         df['IG_value'] = X_flat
         df.to_excel(os.path.join(output_dir_path,"commom_IG_average_absolute_shap.xlsx"),index=False)
         print("snp number",len(X_flat))

    if dataset_dir_path:
         data_paths = mk_dataset_paths(dataset_dir_path)
         data_loaders_func = mk_data_loaders_single_funcs[model_name]
         data_loaders_func_kwargs = {'data_paths': data_paths, 'batch_size': batch_size}
         if dataset_in_memory:
             data_loaders_func_kwargs['in_memory'] = dataset_in_memory
         if persistent_workers:
             data_loaders_func_kwargs['persistent_workers'] = persistent_workers
         if label_data_id_field_name:
             data_loaders_func_kwargs['label_data_id_field_name'] = label_data_id_field_name
         if label_data_label_field_name:
             data_loaders_func_kwargs['label_data_label_field_name'] = label_data_label_field_name
         data_loaders = data_loaders_func(**data_loaders_func_kwargs)
         load_test_dir = os.path.join(load_test, test_dir_prefix)
         writer = SummaryWriter(log_dir=load_test_dir)
         if data_loaders.get('test', None):
             workflows['test'](device, net,data_loaders['test'],model_name,writer)
         else:
             workflows['test'](device, net,data_loaders['valid'],model_name,writer)
         if model_name =='SNPImageNet':
             print('net.return_only_snp_image',net.multi_modal_model.return_only_snp_image)
             net.multi_modal_model.return_only_snp_image = True
         count=1
         n = 0
         all_weights = []
         mean_gene_baseline = None
         mean_img_baseline = None
         for phase in ['train', 'valid']:
             if phase == 'train':
                 for inputs, labels in data_loaders[phase]:
                     inputs = [each_input.to(device) for each_input in inputs]
                     train_gene_tensors, train_img_tensors = inputs
                     #random_idx = torch.randint(0, train_gene_tensors.shape[0], (1,)).item()
                     #random_gene_baseline = train_gene_tensors[random_idx].unsqueeze(0).to(device)
                     #random_img_baseline = train_img_tensors[random_idx].unsqueeze(0).to(device)
                     #baselines = tuple([random_gene_baseline, random_img_baseline])
                     mean_gene_baseline = torch.mean(train_gene_tensors, dim=0).unsqueeze(0).to(device)
                     mean_img_baseline = torch.mean(train_img_tensors, dim=0).unsqueeze(0).to(device)
                     baselines = tuple([mean_gene_baseline, mean_img_baseline])
                     print("baselines",baselines) 
                     break 
             if phase == 'valid':
                 for inputs, labels in data_loaders[phase]:
                     inputs = [each_input.to(device) for each_input in inputs]
                     gene_tensors, img_tensors = inputs                   
                     for i in range(len(gene_tensors)):
                         if n==0:   
                             total_weight_sum=np.zeros(gene_tensors.shape[1])
                         gene_tensor = gene_tensors[i]
                         img_tensor = img_tensors[i]
                         #print(gene_tensor.shape)
                         #print(img_tensor.shape)
                         attributions = make_gene_image_saliency_maps(net, device,
                                                                     baselines=baselines,
                                                                     img_tensor=img_tensor,
                                                                     gene_tensor=gene_tensor,
                                                                     saliency_maps_name=saliency_maps_name,
                                                                     baseline_method=baseline_method)
                         gene_attribution, image_attribution = attributions[0], attributions[1]
                         #print("gene_attribution",gene_attribution)
                         #print("gene_attribution shape:", gene_attribution.shape)
                         weight = np.abs(gene_attribution.flatten().cpu().numpy())#
                         #print("weight",weight)
                         #print("weight shape:", weight.shape)
                         total_weight_sum +=weight  
                         n+=1                 
                         img_saliency_maps_show(image_attribution, img_tensor, output_dir_path,image_file_path, saliency_maps_name,count)
                         count+=1                
         print("samples",n)
         average_weight = total_weight_sum / n
         X_reshape = average_weight.reshape(1, average_weight.shape[0] // 4, 4)
         X_window_sum = np.sum(X_reshape, axis=2)
         X_window_sum = X_window_sum / 4
         X_flat = X_window_sum.flatten()
         print("mean result：",X_flat)
         df = pd.DataFrame()
         df['IG_value'] = X_flat
         df.to_excel(os.path.join(output_dir_path,'SNP',"commom_average_absolute_shap.xlsx"),index=False)
         print("snp number",len(X_flat))
         


if __name__ == '__main__':
    main(test_dir_prefix=datetime_now_str(),
      model_name='SNPImageNet',
      wts_path=r"/pub/data/gaoss/New_Multi/code/work_dirs/records/checkpoints/20241219214429/best_model_checkpoints.pth",#"/pub/data/gaoss/New_Multi/code/work_dirs/records/checkpoints/20241216094657/best_model_checkpoints.pth",
      snp_numbers=100760,#179712
      label_data_id_field_name="学籍号", #participant_id   学籍号
      label_data_label_field_name='high_myopia', #图像和多模态high_myopia，遗传label
      dataset_dir_path=r"/pub/data/gaoss/New_Multi/code/SuLabCohort/20240918215106/",#r"/pub/data/gaoss/New_Multi/code/SuLabCohort/20240918215106/",
      image_file_path='',#r"/pub/data/gaoss/New_Multi/code/SuLabCohort/20240914111247/test/image/2001070033_OS_1010.jpg",
      gene_file_path='',#r"/pub/data/gaoss/New_Multi/code/SuLabCohort/20240918215106/",   
      output_dir_path=r"/pub/data/gaoss/New_Multi/code/IG_Explainability/",   
      saliency_maps_name='IntegratedGradients',  
      baseline_method='gaussian_blur',
      cam_name='GradCAM',
    )