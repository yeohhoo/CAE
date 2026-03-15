# Semantic Editing of Category-Level Latent Representations for Few-Shot Image Generation

In this paper, we propose **Category-Level Attribute Editing (CAE)**,
a method that disentangles and manipulates category-level latent representations for flexible and efficient few-shot synthesis. Our approach adopts a structured two-stage paradigm:
first, a shared latent representation for novel categories is
learned using a pre-trained generative model and lightweight
mapping; second, CLIP-guided latent direction discovery
with region constraints is introduced for semantic editing,
capturing intra-class variations while preserving structural
consistency. This design ensures semantic controllability and
boosts diversity without altering the pre-trained generator.


## Description
Official implementation of CAE for few-shot image generation. Our code is modified from [pSp](https://github.com/eladrich/pixel2style2pixel.git).

## 📁 Project Structure

```bash
├── .vscode/             # the configuration files to automatic format code 
├── criteria/            # 
├── configs/             # the path of dataset
├── datasets/            # preparing the data with episode paradigm
├── environment/         # conda environment file
├── example/             # test 
├── filelists/           # split dataset for train and test
├── models/              # model architecture
├── op/                  # some layer for stylegan2
├── options/             # Parameter definitions used during training, inference
├── pretrained_models/   # the directory of pretrained models
├── scripts/             # Main entry for training, inference, evaluation
├── training/            # Main function for training, inference
├── utils/               # Tool functions for logging, visualization
├── Visualization/      
├── .flake8              # the configuration files of flake8
├── pyproject.toml       # the configuration files of black and isort
└── README.md
```
## ⚙️ Installation
### Prerequisites
- Linux
- NVIDIA GeForce RTX 4090 + CUDA 12.1 + cuDNN 8.9.0
- Python 3
### Clone repository
```bash
git clone https://github.com/yeohhoo/CAE.git
cd CAE
```
 ### Create environment and Install dependencies
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). All dependencies for defining the environment are provided in `environment/cae_env.yml`. You can use the following:

```bash
conda env create -f environment/cae_env.yml
conda activate cae_env
```


## 🚀 Quick Start
  ### Pretrained Model 
You can either download the pre-trained models provide from [path]() and put the pre-trained models under `CAE/pretrained_models`, or train the model on your own dataset.

## Training

### Preparing your Data
+ You should first download your private dataset and organize the file structure as follows:
```
└── data_root
    ├── train                      
    |   ├── cate-id                                   # train-class
    |   |    ├── cate-id_sample-id.jpg                # train-img
    |   |    └── ...                                  # ...
    |   └── ...                                       # ...
    └── test                      
        ├── cate-id                                   # test-class
        |    ├── cate-id_sample-id.jpg                # test-img
        |    └── ...                                  # ...
        └── ...                                       # ...
```
* Here, we provide organized `102flowers` dataset as an example:
```
└── data_root
    ├── train
    |   ├── 0
    |   |    ├── image_06734.jpg
    |   |    └── ...
    |   └── ...
    └── test
        ├── 1
        |    ├── image_05087.jpg
        |    └── ...
        └── ...
```
You can split the dataset and configure their paths through `.filelists/dataset/write_dataset_filelist.py` and `configs/paths_config.py` respectively.

### Train
Go to the path to CAE:
```bash
cd /PATH/CAE/
```
Then you can train the model using:
```
python scripts/train.py --dataset_type=flowers_encode \
--psp_checkpoint_path=pretrained_models/psp_flowers.pt \
--gan_checkpoint_path=pretrained_models/340000.pt \
--exp_dir=log \
--feature_size=512 \
--workers=4 \
--batch_size=4 \
--test_batch_size=4 \
--test_workers=4 \
--val_interval=80000 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=1 \
--l2_lambda=1 \
--image_interval=1000 \
--method=relationnet_PSM \
--train_n_way=4 \
--test_n_way=4 \
--n_eposide=100 \
--metric_file=pretrained_models/best_model.tar \
--learning_rate=5e-5
```
Note: We adopt a two-stage training strategy to stabilize and enhance the LA module. Firstly, train the LA module in the latent space to reconstruct latent codes, preventing destructive impact on the original generator. And then introduce the metric module to guide LA in learning category-level semantic features.

### Inference
Having trained your model or using pre-trained models we provide, you can use scripts/inference_clip.py to apply the model to generate more samples.

```
python scripts/inference_clip.py --dataset_type=flowers_encode \
--checkpoint_path=/root/c_1206/CAE/log_1/checkpoints/iteration_20000.pt \
--exp_dir=/root/c_1206/CAE/log_test \
--feature_size=512 \
--workers=4 \
--batch_size=4 \
--test_batch_size=4 \
--test_workers=4 \
--val_interval=80000 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=1 \
--l2_lambda=1 \
--image_interval=1000 \
--hyperbolic_lambda=0.3 \
--reverse_lambda=1 \
--method=relationnet_PSM \
--train_n_way=4 \
--test_n_way=4 \
--n_eposide=100 \
--metric_file=/root/c_1206/CAE/pretrained_models/best_model.tar \
--learning_rate=5e-5 \
--image_list=/root/c_1206/CAE/example/test.list
```


### Evaluate
You can evaluate the generated samples using the script we provided in `scripts/calc_metric.py`. We also provide our [generated samples](https://drive.google.com/drive/folders/1gx7Vx7kvGa78taePSJ_wrkiyZrYPRwNI?usp=sharing) by the model we have trained.
