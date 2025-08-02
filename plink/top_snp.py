
import numpy as np
import pandas as pd
import os
import tqdm
def choose_top_snp(input_dir_path,top_shap_path,output_path):
    npy_file_names = [file_name for file_name in os.listdir(input_dir_path)]
    last_layer = os.path.basename(os.path.normpath(input_dir_path))
    output1 = os.path.join(output_path,last_layer)
    print(output1)

    df = pd.read_excel(top_shap_path)
    snp_index = df['row.names'] - 1
    snp_index=snp_index.tolist()
    snp_index = sorted(snp_index, reverse=False)
    print("top snp number:",len(snp_index))

    try:
        os.makedirs(output1, exist_ok=True)
    except OSError as e:
        print(f"Error creating directories: {e}")

    for i in tqdm.tqdm(range(len(npy_file_names))):
        one_hot = np.load(os.path.join(input_dir_path, npy_file_names[i]))
        extracted_top = one_hot[:, snp_index]
        output = os.path.join(output1, npy_file_names[i])
        np.save(output,extracted_top)



if __name__ == '__main__':
    input_dir_path = r"/pub/data/gaoss/New_Multi/code/One-hot/no-delect-no-rs/valid-one-hot/"
    top_shap_path = r"/pub/data/gaoss/New_Multi/code/SHAP/20240821211012/MAGIC_top_snps0.0005.xlsx"
    output_path = r"/pub/data/gaoss/New_Multi/code/valid_contribution/shap0.0005/top-one-hot/"
    choose_top_snp(input_dir_path,top_shap_path,output_path)





