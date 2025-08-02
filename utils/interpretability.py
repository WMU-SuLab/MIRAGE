

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_grad_cam as grad_cams
import torch
from matplotlib import colors
from torch import nn
import torchcam.methods as torch_cams
from captum.attr import Saliency, IntegratedGradients, NoiseTunnel, visualization as viz
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image, gaussian_blur
from matplotlib import cm
from utils.image.noise import image_random_noise, image_gaussian_noise
from .image import tensor2numpy
matplotlib.use('agg')
#matplotlib.use('Agg') 
def imshow(images, titles=None, file_name="test.jpg", dir_path: str = './', size=6):
    lens = len(images)
    fig = plt.figure(figsize=(size * lens, size))
    if not titles:
        titles = "0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(images[i - 1].shape) == 2:
            plt.imshow(images[i - 1], cmap='Reds')
        else:
            plt.imshow(images[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(os.path.join(dir_path, f'{file_name}'), bbox_inches='tight')


class ReshapeTransform:
    def __init__(self, height: int = 14, width: int = 14):
        self.height = height
        self.width = width

    def __call__(self, tensor):
        result = tensor.reshape(tensor.size(0), self.height, self.width, tensor.size(2))

        # Bring the channels to the first dimension, like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result


def make_img_cam(net, cam_name, target_layers: list, targets: list = None, img_tensor=None,
                 use_cuda=False):

    if not img_tensor:
        raise ValueError('please input img tensor or gene tensor')
    input_tensor = img_tensor.unsqueeze(0)

    if CAM := getattr(grad_cams, cam_name, None):
        cam = CAM(model=net, target_layers=target_layers, use_cuda=use_cuda)
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
        # for grayscale_cam in grayscale_cams:
        grayscale_cam = grayscale_cams[0, :]
        img = tensor2numpy(img_tensor)
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        return img, grayscale_cam, visualization
    elif CAM := getattr(torch_cams, cam_name, None):
        with CAM(net, target_layers) as cam_extractor:
            # Preprocess your data and feed it to the model
            out = net(input_tensor)
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        img = tensor2numpy(img_tensor)
        visualization = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'),
                                     alpha=0.5)
        return img, activation_map[0].squeeze(0).numpy(), visualization
    else:
        raise ValueError(f'{cam_name} does not exist.')


def img_saliency_maps_show(image_attribution, img_tensor, output_dir_path,image_file_path, saliency_maps_name,count):
    image_attribution_norm = np.transpose(image_attribution.detach().cpu().squeeze().numpy(), (1, 2, 0)) 
    cmap = cm.get_cmap('jet')
    # 可视化 IG 值
    if image_file_path!='':
        file_name = os.path.basename(image_file_path)
    else:
        file_name=count
    print('count',file_name)
    plt_fig, plt_axis = viz.visualize_image_attr_multiple(image_attribution_norm,  # 224,224,3
                                                              tensor2numpy(img_tensor),  # 224,224,3
                                                              # np.transpose(img_tensor.squeeze().cpu().detach().numpy(),
                                                              #              (1, 2, 0)),
                                                              methods=["original_image","blended_heat_map","heat_map"],
                                                              signs=["all","absolute_value","absolute_value" ],
                                                              fig_size=(16, 6),
                                                              # cmap=default_cmap,
                                                              cmap=cmap,
                                                              show_colorbar=True,
                                                              alpha_overlay=0.5,
                                                              outlier_perc=20)
    plt_fig.savefig(os.path.join(output_dir_path,'Image',  f'Dark_blue_heat_map{file_name}.pdf'), bbox_inches='tight')
    plt.close(plt_fig)
    plt_fig, plt_axis = viz.visualize_image_attr_multiple(image_attribution_norm,
                                                          tensor2numpy(img_tensor),
                                                          methods=["original_image","blended_heat_map","heat_map"],
                                                          signs=["all","absolute_value","absolute_value" ],
                                                          fig_size=(16, 6),
                                                          cmap=plt.cm.Blues,
                                                          show_colorbar=True,
                                                          alpha_overlay=0.5,
                                                          outlier_perc=20)

    plt_fig.savefig(os.path.join(output_dir_path,'Image',  f'Light_blue_ink_{file_name}.pdf'), bbox_inches='tight')
    plt.close(plt_fig)

    plt_fig, plt_axis = viz.visualize_image_attr_multiple(image_attribution_norm,  # 224,224,3
                                                          tensor2numpy(img_tensor),  # 224,224,3
                                                          methods=["original_image","blended_heat_map","heat_map"],
                                                          signs=["all","absolute_value","absolute_value" ],
                                                          fig_size=(16, 6),
                                                          #cmap=default_cmap,
                                                          cmap=plt.cm.inferno,
                                                          show_colorbar=True,
                                                          alpha_overlay=0.5,
                                                          outlier_perc=20)
    plt_fig.savefig(os.path.join(output_dir_path,'Image',  f'Purple_and_black_image_{file_name}.pdf'),bbox_inches='tight')
    plt.close(plt_fig)
    plt_fig, plt_axis = viz.visualize_image_attr_multiple(image_attribution_norm,
                                                          tensor2numpy(img_tensor),
                                                          methods=["original_image","blended_heat_map","heat_map"],
                                                          signs=["all","absolute_value","absolute_value" ],
                                                          fig_size=(16, 6),
                                                          cmap=plt.cm.viridis,
                                                          show_colorbar=True,
                                                          alpha_overlay=0.5,
                                                          outlier_perc=20)
    plt_fig.savefig(os.path.join(output_dir_path,'Image', f'Green_gradient_color_map_{file_name}.pdf'),bbox_inches='tight')
    plt.close(plt_fig)
   

def make_saliency_maps(net, device,baselines, inputs: tuple[torch.Tensor, ...], saliency_maps_name: str = 'IntegratedGradients',
                       baseline_method: str = 'gaussian_blur'):
    inputs_unsqueezed = tuple([item.unsqueeze(0) for item in inputs])

    net.eval()
    
    if saliency_maps_name == 'Saliency':
        sa = Saliency(net)
        attributions = sa.attribute(inputs, target=0, abs=False)
    elif saliency_maps_name == 'IntegratedGradients':
        #print(baselines)
        ig = IntegratedGradients(net)
        attributions, delta = ig.attribute(inputs_unsqueezed, baselines, target=0, n_steps=100,return_convergence_delta=True)
        #print('IG Attributions:', attributions)
        print('Convergence Delta:', delta)
        # return attributions, delta
        # return attributions, delta, baselines
    elif saliency_maps_name == 'NoiseTunnel':
        ig = IntegratedGradients(net)
        noise_tunnel = NoiseTunnel(ig)
        attributions = noise_tunnel.attribute(inputs, target=0, nt_samples=12, nt_type='smoothgrad_sq', )
    else:
        raise ValueError('no this saliency maps method')
    return attributions


def make_image_saliency_maps(net, device, image_tensor: torch.Tensor, saliency_maps_name, baseline_method):
    return make_saliency_maps(net, device, (image_tensor,), saliency_maps_name, baseline_method)

def make_gene_saliency_maps(net, device, gene_tensor: torch.Tensor, saliency_maps_name, baseline_method):
    return make_saliency_maps(net, device, (gene_tensor,), saliency_maps_name, baseline_method)


def make_gene_image_saliency_maps(net, device,baselines, gene_tensor: torch.Tensor, img_tensor: torch.Tensor, saliency_maps_name,
                                  baseline_method, ):
    return make_saliency_maps(net, device,baselines, (gene_tensor, img_tensor), saliency_maps_name, baseline_method)
