
import os
import numpy as np
import pandas as pd

def Test_Aver_Absolute_Shap(path:str,shap_path:str):
    X_loaded = np.load(path)
    print(X_loaded.shape)
    X_loaded = np.abs(X_loaded)
    X_sum = np.sum(X_loaded, axis=0, keepdims=True)
    X_sum = X_sum / X_loaded.shape[0]
    X_reshape = X_sum.reshape(1, X_loaded.shape[1] // 4, 4)
    X_window_sum = np.sum(X_reshape, axis=2)
    print(X_window_sum.shape)
    X_window_sum = X_window_sum / 4
    X_flat = X_window_sum.flatten()
    print("Mean calculation results：",X_flat)
    df = pd.DataFrame()
    df['shap_value'] = X_flat
    df.to_excel(os.path.join(shap_path,"MAGIC_average_absolute_shap.xlsx"),index=False)
    print("snp number",len(X_flat))

def Valid_Aver_Absolute_Shap(npy_file_names:list,shap_path:str):
    sum_flat = None
    file_count = 0
    for i in range(len(npy_file_names)):
        X_loaded = np.load(os.path.join(shap_path, npy_file_names[i]))
        X_loaded = np.abs(X_loaded)
        X_sum = np.sum(X_loaded, axis=0, keepdims=True)
        X_sum = X_sum / X_loaded.shape[0]
        X_reshape = X_sum.reshape(1, X_loaded.shape[1] // 4, 4)
        X_window_sum = np.sum(X_reshape, axis=2)
        X_window_sum = X_window_sum / 4
        X_flat = X_window_sum.flatten()
        if sum_flat is None:
            sum_flat = X_flat
        else:
            sum_flat += X_flat

        file_count += 1

    if file_count > 0:
        mean_flat = sum_flat / file_count
        df = pd.DataFrame()
        df['shap_value'] = mean_flat
        df.to_excel(os.path.join(shap_path,"MAGIC_average_absolute_shap.xlsx"),index=False)
        print("Mean calculation results：", mean_flat)
        print("snp number",len(mean_flat))
    else:
        print("No files to process")



#shap_path =r"/pub/data/gaoss/New_Multi/code/SHAP/20240814163543/"
#path=r"/pub/data/gaoss/New_Multi/code/SHAP/20240814163543/MAGIC_shap_value.npy"
#Aver_Absolute_Shap(path,shap_path)#平均绝对shap_value
