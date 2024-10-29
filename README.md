# agroforest-forest mapping in Peru
## code for agroforestry mapping in peru

## How to use the code

### 0 set the conda environment
conda env create -f environment_for_agroforest.yml

### 1 prepare the data

#### 1.1 Download the samples within the link (https://zenodo.org/records/13946752)
#### 1.2 Download the Nicfi tiles overlay with the samples
#### 1.3 Stack the Nicfi data with samples to make the data ready for training
##### we saved the training dataset with .npy

### 2 training

#### 2.1 change the dataroot with your dataset
#### 2.2 ready for training
python train.py --model unet --alpha 0.1 --batch-size 16 --gpu-ids 0 --imsize 128 --lr 8e-7 --lr-policy poly --net resnet50 --num-threads 32 --niter 1000 --val-freq 2000 --print-freq 10 --checkpoints-dir your_pathway_for_model
