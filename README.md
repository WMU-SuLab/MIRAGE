
<!-- README.md is generated from README.Rmd. Please edit that file -->

# MIRAGE

<!-- badges: start -->
<!-- badges: end -->

**MIRAGE (Multimodal Interpretable Risk Assessment of Genetic and Eye-imaging data)** is a deep learning framework designed for personalized risk prediction by integrating exome-wide genotype data with fundus image analysis.

The MIRAGE framework requires paired WES (Whole Exome Sequencing) data and fundus image as input. The model combines DeepExGRS for genetic risk modeling and a convolutional network for imaging, fused via a gating attention mechanism. See below for the steps to train the data or use the model generated in our paper.

![image](figures/MIRAGE.png)

## Tutorials

### Train a MIRAGE model based on test datasets

Example input files are provided in the folder `base_code/example/0108`.

#### Arguments

``` r
###Train a MIRAGE model by "SNPImageNet"
python train.py --train_dir_prefix "test_training" \
--model_name "SNPImageNet" \
--dataset_dir_path "base_code/example/0108" \
--snp_number 100760 \
--batch_size 4 

####For "SNPNet": python train.py --train_dir_prefix 'test_training_SNP' 
#--model_name "SNPNet" \
#--dataset_dir_path "base_code/example/0108" \
#--snp_number 100760 --batch_size 4 --label_data_label_field_name "label"
####For "ImageNet":  python train.py --train_dir_prefix 'test_training_image' \
#--model_name "ImageNet" \
#--dataset_dir_path "/base_code/example/0108" \
#--batch_size 4 \
#--label_data_label_field_name "high_myopia"

###############Results#################
GPU is available： True
CUDA is available!
GPU numbers: 1
CUDA version: 11.8
current device:cuda:0
SNPImageNet
kerel_width 16
fc_0_kernel_size 16
kerel_width 16
fc_0_kernel_size 16
model_checkpoints /share/pub/MIRAGE/base_code/work_dirs/records/checkpoints/test_training
/share/pub/MIRAGE/base_code/work_dirs/records/checkpoints/test_training/best_model_checkpoints.pth
进行训练 train
matrix [[ 6 41]
 [ 6 63]]
Epoch 1/200 | train | Loss: 3467.4295 | best F1: 0
acc: 0.5948 | precision: 0.6058 | recall: 0.9130 | 
f1: 0.7283 | mcc: 0.0656 | auc: 0.5720  | 
tpr: 0.9130 | fpr: 0.8723 | ks: 0.0407 | sp: 0.1277
Time: 0m 40s
进行训练 valid
matrix [[ 0  6]
 [ 0 22]]
Epoch 1/200 | valid | Loss: 3184.5612 | best F1: 0.88
...
...
...
进行训练 train
matrix [[44  0]
 [ 0 72]]
Epoch 200/200 | train | Loss: 1375.2559 | best F1: 0.9743589743589743
acc: 1.0000 | precision: 1.0000 | recall: 1.0000 | 
f1: 1.0000 | mcc: 1.0000 | auc: 1.0000  | 
tpr: 1.0000 | fpr: 0.0000 | ks: 1.0000 | sp: 1.0000
Time: 15m 24s
进行训练 valid
matrix [[ 3  4]
 [ 0 21]]
Epoch 200/200 | valid | Loss: 1274.1423 | best F1: 0.9743589743589743
acc: 0.8571 | precision: 0.8400 | recall: 1.0000 | 
f1: 0.9130 | mcc: 0.6000 | auc: 0.9422  | 
tpr: 1.0000 | fpr: 0.5714 | ks: 0.4286 | sp: 0.4286
Time: 15m 25s
Training complete in 15m 25s
Best val f1: 0.974359
matrix [[10  0]
 [ 2  4]]
Test Metrics: 
ACC: 0.8750 | Precision: 1.0000 | Recall: 0.6667 | 
F1: 0.8000 | MCC: 0.7454 | AUC: 0.9667 | 
tpr: 0.6667 | fpr: 0.0000 | KS: 0.8333 | SP: 1.0000
```

### Test results by trained model

Example input files are provided in the folder `base_code/example/0108`. \
Model trained by example input files will in `"base_code/work_dirs/records/checkpoints/test_training/best_model_checkpoints.pth"`.

#### Arguments

``` r
#Test results by trained model
python test.py --test_dir_prefix "test_MIRAGE" \
--model_name "SNPImageNet" \
--dataset_dir_path "base_code/example/0108" \
--wts_path "base_code/work_dirs/records/checkpoints/test_training/best_model_checkpoints.pth" \
--snp_number 100760 \
--batch_size 4

###############Results#################
Output_folder: test_MIRAGE
SNPImageNet
kerel_width 16
fc_0_kernel_size 16
kerel_width 16
fc_0_kernel_size 16
load_test_dir:  /share/pub/MIRAGE/base_code/work_dirs/records/load_test/test_MIRAGE
##############Start test################
matrix [[10  0]
 [ 2  4]]
Test Metrics: 
ACC: 0.8750 | Precision: 1.0000 | Recall: 0.6667 | 
F1: 0.8000 | MCC: 0.7454 | AUC: 0.9667 | 
tpr: 0.6667 | fpr: 0.0000 | KS: 0.8333 | SP: 1.0000
```
| Argument | Description |
| ---------- | ----------- |
| __--train_dir_prefix__ | Parameter **specific** to `train.py`. Output folder for results under "base_code/work_dirs/records/checkpoints". |
| __--test_dir_prefix__ |  Parameter **specific** to `test.py`. Output folder for results under "base_code/work_dirs/records/checkpoints". |
| __--model_name__ | Model selection among SNPImageNet, SNPNet and ImageNet (default: SNPImageNet). |
| __--dataset_dir_path__ | Input folder of datasets, contaning `test`, `train`, and `valid` folders. For each datasets folder, it must contain a `gene` folder (holding the WES one-hot encoded data for all samples), an `image` folder (containing the fundus images for all samples), and a table named `labels.csv` with meta information for all samples. By default, the column names for the sample ID and label are "participant_id" and "high_myopia", respectively. |
| __--wts_path__ |  Parameter **specific** to `test.py`. Trained Model path ,e.g. `MIRAGE/base_code/example/best_model_checkpoints.pth`. |
| __--snp_number__ |  Number of used SNPs for training, e.g. 100760 for `SNPImageNet` and `SNPNet` model. The default value set as **None** for `ImageNet` model.  |
| __--batch_size__ | Batch size for training (default:32). Note: this parameter should be less than the sample size of each used dataset. |
| __--label_data_id_field_name__ | Column name of sample ID (default: "participant_id"). |
| __--label_data_label_field_name__ | Column name of sample label (default: "high_myopia", for SNPImageNet and ImageNet). If the training model is set to "SNPNet", then assign it the value "label". |

















