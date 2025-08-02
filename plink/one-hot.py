import os
import subprocess
from pathlib import Path
from shutil import which
from typing import Generator, Tuple, Sequence
import numpy as np
import torch
from torch.nn.functional import one_hot
from bed_reader import open_bed


def _lines_in_file(file_path: Path) -> int:
    with open(file_path, "r") as f:
        return sum(1 for _ in f)

def validate_npy_files(raw_data_path):
    file_count = 0
    fam_file = next(i for i in Path(str(raw_data_path)).iterdir() if i.suffix == ".fam")
    bim_file = next(i for i in Path(str(raw_data_path)).iterdir() if i.suffix ==  ".bim")
    indiv_sample_size = _lines_in_file(file_path=fam_file)
    snp_sample = _lines_in_file(file_path=bim_file)
    return indiv_sample_size,snp_sample

def ExternalRawData(raw_data_path):
    bed_files = [i for i in Path(str(raw_data_path)).iterdir() if i.suffix == ".bed"]
    if len(bed_files) != 1:
        raise ValueError(
            f"Expected one .bed file in {raw_data_path}, but"
            f"found {bed_files}."
        )

    return str(bed_files[0])
# a=ExternalRawData(raw_data_path)
# print(a)
def PlinkExtractAlleles(raw_data_path,
                        qc_outpath,
                        extract_snp_file=""
                        ):
    plink_input = raw_data_path
    plink_output=qc_outpath

    cmd = [
        "plink",
        "--bfile",
        plink_input,
        "--extract",
        extract_snp_file,
        "--make-bed",
        "--out",
        plink_output,
    ]

    subprocess.call(cmd)
    return os.path.dirname(plink_output)
def _validate_plink_exists_in_path():
    if which("plink") is None:
        raise RuntimeError(
            "plink is not installed or is not in the path. "
            "Please install plink and try again."
        )
def PlinkQC( raw_data_path,
             indiv_sample_size,
             chr_sample,
             qc_output_path,
             extract_snp_output_path='',
             autosome_only=False,
             extract_snp_file=""):
    if extract_snp_file:
        input_path=PlinkExtractAlleles(raw_data_path, extract_snp_file,extract_snp_output_path)
    else:
        input_path=ExternalRawData(raw_data_path)
    # print(f'ExternalRawData',input_path)
    _validate_plink_exists_in_path()
    file_dir=os.path.dirname(input_path)
    # print(file_dir)
    file_prefix = os.path.splitext(os.path.basename(input_path))[0]
    # print(file_prefix)
    plink_input=os.path.join(file_dir, file_prefix)
    # print(plink_input)
    plink_output = qc_output_path
    plink = r"C:\Users\gaoss\plink\plink.exe"
    cmd = [
        plink,
        "--bfile",
        plink_input,
        "--maf",
        str(0.001),
        "--geno",
        str(0.03),
        "--mind",
        str(0.1),
        "--make-bed",
        "--out",
        plink_output,
        "--allow-no-sex"
    ]
    if indiv_sample_size:
        cmd += ["--thin-indiv-count", str(indiv_sample_size)]
    if chr_sample:
        cmd += ["--chr", str(chr_sample)]
    if autosome_only:
        cmd += ["--autosome"]
    print(cmd)
    subprocess.call(cmd)
    return os.path.dirname(plink_output)
# b=PlinkQC(raw_data_path,indiv_sample_size,chr_sample,qc_output_path)
# print(b)
def get_sample_generator_from_bed(
    bed_path: Path,
    chunk_size: int = 1000,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    with open_bed(bed_path) as bed_handle:
        n_samples = bed_handle.iid_count
        for index in range(0, n_samples, chunk_size):
            samples_idx_start = index
            samples_idx_end = index + chunk_size
            ids = bed_handle.iid[samples_idx_start:samples_idx_end]
            arrays = bed_handle.read(
                index=np.s_[samples_idx_start:samples_idx_end, :],
                dtype=np.int8,
            )
            arrays[arrays == -127] = 3     # NA is encoded as -127
            yield ids, arrays
            print("Processed {} samples".format(index + chunk_size))
def _get_one_hot_encoded_generator(
    chunked_sample_generator: Generator[Tuple[Sequence[str], np.ndarray], None, None]
) -> Generator[Tuple[str, np.ndarray], None, None]:
    for id_chunk, array_chunk in chunked_sample_generator:
        array_tensor = torch.from_numpy(array_chunk).to(dtype=torch.long)
        one_hot_encoded = one_hot(array_tensor, num_classes=4)

        # convert (n_samples, n_snps, 4) -> (n_samples, 4, n_snps)
        one_hot_encoded = one_hot_encoded.transpose(2, 1)
        one_hot_encoded = one_hot_encoded.numpy().astype(np.int8)

        assert (one_hot_encoded[0].sum(0) == 1).all()
        assert one_hot_encoded.dtype == np.int8
        for id_, array in zip(id_chunk, one_hot_encoded):
            yield id_, array

def write_one_hot_outputs(
        id_array_generator,
        output_folder):
     _write_one_hot_arrays_to_disk(
            id_array_generator=id_array_generator,
            output_folder=output_folder)

def _write_one_hot_arrays_to_disk(
        id_array_generator,
        output_folder
):
    for id_, array in id_array_generator:
        output_path = os.path.join(output_folder, f"{id_}.npy")
        print(output_path)
        print(id_)
        np.save(str(output_path), array)

def OneHotSNPs(raw_data_path,
               qc,
               qc_output_path,
               OneHotSNPs_output_path,
               autosome_only=False,
               indiv_sample_size=None,
               chr_sample=None,
               extract_snp_file=""):
    base=""
    if qc:
        base= PlinkQC(raw_data_path,indiv_sample_size,chr_sample,qc_output_path)
    if not base:
        base=raw_data_path
    bed_files = [i for i in Path(str(base)).iterdir() if i.suffix == ".bed"]
    input_path = bed_files[0]
    chunk_generator = get_sample_generator_from_bed(
        bed_path=input_path,
        chunk_size=int(array_chunk_size)
    )
    sample_id_one_hot_array_generator = _get_one_hot_encoded_generator(
        chunked_sample_generator=chunk_generator
    )
    write_one_hot_outputs(
        id_array_generator=sample_id_one_hot_array_generator,
        output_folder=  OneHotSNPs_output_path)


if __name__ =="__main__":
    raw_data_path = r"/pub/data/gaoss/data/MAGIC/no-delect-no-rs/MAGIC_prs/75_HM/test/" # help="Path to raw data folder to be processed (containing data.bed, data.fam,data.bim)
    extract_snp_output_path = None
    qc_output_path = None
    OneHotSNPs_output_path = r"/pub/data/gaoss/New_Multi/code/MAGIC_prs_one_hot/75_HM/test-one-hot/"
    output_folder = ""  # help="Folder to save the processed data in."
    qc = False  # help="Whether to do basic QC on plink data (--maf 0.001, --geno 0.03, --mind 0.1). Default: False."
    array_chunk_size = 1000  # help="How many individuals to process at a time. " "Useful to avoid running out of memory.",
    indiv_sample_size = None  # help="How many individuals to randomly sample." " Only applicable if do_qc is set."
    chr_sample = None  # help="Which chromosomes to sample, follows plink notation." " Only applicable if do_qc is set."
    snp_sample = None
    autosome_only = False  # help="Whether to only use autosomes. " "Only applicable if do_qc is set."
    extract_snp_file = ""  # help=".bim file to use if generating only the "  "intersection between the data and the "  "specified .bim file."
    OneHotSNPs(raw_data_path, qc, qc_output_path, OneHotSNPs_output_path)
    indiv_sample_size, snp_sample = validate_npy_files(raw_data_path)
    print(f'indiv_sample_size',indiv_sample_size, f'snp_sample',snp_sample)
    print('finish')





