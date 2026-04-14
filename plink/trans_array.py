
import numpy as np
import os
import tqdm
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, 8192))
import argparse

def convert(npy_file_names,output_dir_path):
    participant_files = {
        participant_id.split(".")[0]: open(os.path.join(output_dir_path, f'{participant_id.split(".")[0]}.csv'), 'w')
        for participant_id in npy_file_names}
    for participant_id, participant_file in participant_files.items():
        participant_file.write(f'{participant_id},')
    for csv_file_name in tqdm.tqdm(
            npy_file_names):
        one_hot = np.load(os.path.join(input_dir_path, csv_file_name))
        # print(one_hot)
        # print(one_hot.shape)
        # print(one_hot.T.reshape(1, -1))
        one_hot = one_hot.T.reshape(1, -1)
        # print(one_hot.T.reshape(1, -1).shape)
        participant_id = csv_file_name.split(".")[0]
        participant_file = participant_files[participant_id]
        participant_file.write(','.join(map(str, one_hot[0])))
        participant_file.close()

parser = argparse.ArgumentParser(description='Trans one-hot results from .npy to .csv')
parser.add_argument('--input_dir_path', required=True,default="./one-hot_npy", help='Input folder for one-hot .npy results')
parser.add_argument('--output_dir_path', required=False,default="./one-hot_results", help='Output folder for final one-hot results')

args = parser.parse_args()

if __name__ =="__main__":
    input_dir_path = args.input_dir_path
    output_dir_path = args.output_dir_path
    npy_file_names = [file_name for file_name in os.listdir(input_dir_path)]
    print(len(npy_file_names))
    if len(npy_file_names)>8000:
        participant_first = npy_file_names[:8000]
        print(len(participant_first))
        participant_last = npy_file_names[8000:]
        print(len(participant_last))
        convert(participant_first, output_dir_path)
        convert(participant_last, output_dir_path)
    else:
        convert(npy_file_names, output_dir_path)











