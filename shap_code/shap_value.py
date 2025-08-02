import numpy as np
import pandas as pd
import shap
import os
import random
import torch
import sys
sys.path.append("/pub/data/gaoss/New_Multi/code")
from utils.dir import mk_dir
import matplotlib.pyplot as plt
from captum.attr import Saliency, IntegratedGradients, visualization as viz 
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from shap_code.average_absolute_SHAP import Test_Aver_Absolute_Shap,Valid_Aver_Absolute_Shap
def SNP_Shap_Value(net,background,dataset,device,test_dir_prefix,valid=False):
    global explainer
    if valid==False:   
        print("Calculate the test set shap value:")
        output_path = r"/pub/data/gaoss/New_Multi/code/SHAP/"
        i=0
        shape_values = []
        for inputs,_ in background:
            valid = [each_input.to(device) for each_input in inputs]
            explainer = shap.DeepExplainer(net, *valid)
            break  
        for inputs, _ in dataset:
            test = [each_input.to(device) for each_input in inputs]
            shap_value = explainer.shap_values(*test, check_additivity=False)
            print(f'{i}',shap_value)
            print(len(shap_value))
            shape_values.append(shap_value)
        SHAP_value = [data.squeeze(axis=2) for data in shape_values]
        X = np.concatenate(SHAP_value, axis=0)
        shap_path = os.path.join(output_path, test_dir_prefix)
        mk_dir(shap_path)
        path=os.path.join(shap_path,"MAGIC_shap_value.npy")
        np.save(path, X)
        print("Calculate the mean absolute value")
        Test_Aver_Absolute_Shap(path,shap_path)
    else:
        print("Calculate the validation set shap value:")
        output_path = r"/pub/data/gaoss/New_Multi/code/SHAP/"
        shap_path = os.path.join(output_path, test_dir_prefix)
        mk_dir(shap_path)
        i = 0
        j = 0
        shape_values = []
        for inputs, _ in background:
            valid = [each_input.to(device) for each_input in inputs]
            explainer = shap.DeepExplainer(net, *valid)
            break
        for inputs, _ in dataset:
            i += 1
            test = [each_input.to(device) for each_input in inputs]
            shap_value = explainer.shap_values(*test, check_additivity=False)
            #print(f'{i}', shap_value)
            #print(len(shap_value))
            shape_values.append(shap_value)
            if len(shape_values) > 36:
                j += 1
                path = os.path.join(shap_path, f"MAGIC_shap_value{j}.npy")
                SHAP_value = [data.squeeze(axis=2) for data in shape_values]
                X = np.concatenate(SHAP_value, axis=0)
                np.save(path, X)
                shape_values = []
        if len(shape_values) != 0:
            j += 1
            path = os.path.join(shap_path, f"MAGIC_shap_value{j}.npy")
            SHAP_value = [data.squeeze(axis=2) for data in shape_values]
            X = np.concatenate(SHAP_value, axis=0)
            np.save(path, X)

        npy_file_names = [file_name for file_name in os.listdir(shap_path)]
        print("npy_file_names",npy_file_names)
        print("Calculate the mean absolute value")
        Valid_Aver_Absolute_Shap(npy_file_names, shap_path)


        
    
    
        
    

        





  

