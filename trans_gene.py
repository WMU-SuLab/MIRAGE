import numpy as np
import os
import tqdm
import resource
from utils.dir import mk_dir
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, 8192))
def convert(path,npy_file_names,output_dir_path):
    participant_files = {
        participant_id.split(".")[0]: open(os.path.join(output_dir_path, f'{participant_id.split(".")[0]}.csv'), 'w')
        for participant_id in npy_file_names}
    for participant_id, participant_file in participant_files.items():
        participant_file.write(f'{participant_id},')
    for csv_file_name in tqdm.tqdm(
            npy_file_names):
        one_hot = np.load(os.path.join(path, csv_file_name))
        # print(one_hot)
        # print(one_hot.shape)
        # print(one_hot.T.reshape(1, -1))
        one_hot = one_hot.T.reshape(1, -1)
        # print(one_hot.T.reshape(1, -1).shape)
        participant_id = csv_file_name.split(".")[0]
        participant_file = participant_files[participant_id]
        participant_file.write(','.join(map(str, one_hot[0])))
        participant_file.close()

if __name__ =="__main__":
    gene_list = r"/pub/data/gaoss/data/MAGIC/no-delect-no-rs/MAGIC_prs/75_HM_gene/gene_name.txt"
    input_dir_path = r"/pub/data/gaoss/New_Multi/code/MAGIC_prs_one_hot/75_HM_gene/test-one-hot/"
    output_dir_path = r"/pub/data/gaoss/New_Multi/code/MAGIC_prs_one_hot/75_HM_gene/conver_test/"
    with open(gene_list, "r") as f:
        genes = f.read().splitlines()
    i=0
    for gene in tqdm.tqdm(genes):
        i+=1
        path=os.path.join(input_dir_path,gene)
        outpath=os.path.join(output_dir_path,gene)
        mk_dir(outpath)
        npy_file_names = [file_name for file_name in os.listdir(path)]
        print(len(npy_file_names))
        if len(npy_file_names)>8000:
            participant_first = npy_file_names[:8000]
            print(len(participant_first))
            participant_last = npy_file_names[8000:]
            print(len(participant_last))
            convert(path,participant_first, output_dir_path)
            convert(path,participant_last, output_dir_path)
        else:
            convert(path,npy_file_names, outpath)
    print(f"{i} genes complete")