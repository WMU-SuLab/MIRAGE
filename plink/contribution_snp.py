
import numpy as np
import pandas as pd
import os
import tqdm
def choose_top_snp(input_dir_path,top_shap_path,output_path):
    npy_file_names = [file_name for file_name in os.listdir(input_dir_path)]
    print("file numbers:",len(npy_file_names))
    df = pd.read_excel(top_shap_path)
    snp_index = df['row.names'] - 1
    snp_index=snp_index.tolist()
    last_layer = os.path.basename(os.path.normpath(input_dir_path))
    output1 = os.path.join(output_path, "top-to(-1)",last_layer)
    output2 = os.path.join(output_path, "other-to(-1)",last_layer)

    try:
        os.makedirs(output1, exist_ok=True)
        os.makedirs(output2, exist_ok=True)
    except OSError as e:
        print(f"Error creating directories: {e}")


    print("top one-hot to -1")
    for i in tqdm.tqdm(range(len(npy_file_names))):
        one_hot = np.load(os.path.join(input_dir_path, npy_file_names[i]))
        for j in snp_index:
            one_hot[:,j]=-1
        output = os.path.join(output1, npy_file_names[i])
        np.save(output, one_hot)

    print("other one-hot to -1")
    for i in tqdm.tqdm(range(len(npy_file_names))):
        one_hot = np.load(os.path.join(input_dir_path, npy_file_names[i]))
        num_cols=one_hot.shape[1]
        for col in range(num_cols):
            if col in snp_index:
                continue
            else:
                one_hot[:, col] = -1
        output = os.path.join(output2, npy_file_names[i])
        np.save(output, one_hot)


if __name__ == '__main__':
    input_dir_path = r"/pub/data/gaoss/New_Multi/code/One-hot/no-delect-no-rs/test-one-hot/"
    top_shap_path = r"/pub/data/gaoss/New_Multi/code/SHAP/20240821211012/MAGIC_top_snps0.0008.xlsx"
    output_path = r"/pub/data/gaoss/New_Multi/code/valid_contribution/shap0.0008/"
    choose_top_snp(input_dir_path,top_shap_path,output_path)