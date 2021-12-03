# InsCLR: Improving Instance Retrieval with Self-Supervision

This is an official PyTorch implementation of the [InsCLR paper](https://arxiv.org/abs/2112.01390).

## Download Dataset
|Dataset|Image|Annotation|
|:-:|:-:|:-:|
|The Oxford Buildings Dataset|[download](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)|[download](http://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/gnd_roxford5k.pkl)|
|The Paris Dataset|[download](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)|[download](http://cmp.felk.cvut.cz/revisitop/data/datasets/rparis6k/gnd_rparis6k.pkl)|
|Google Landmarks Dataset v2|[download](https://github.com/cvdfoundation/google-landmark/blob/master/download-dataset.sh)|[download](https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv)|
|R-1M distractor|[download](http://ptak.felk.cvut.cz/revisitop/revisitop1m/)||
|INSTRE|[download](http://123.57.42.89/Dataset_ict/INSTRE/INSTRE_release.rar)|[download](https://cmp.felk.cvut.cz/~iscenahm/files/test_script.zip)|

We also provide scripts for downloading these datasets (see download).

## Training
To meet the performance reported in the paper, you need several training stages, and each training stage may have a different config, but they share a common pipeline.

Generally, a training stage includes the following steps.

### Extract Features
Using backbone pretrained on ImageNet or trained on previous stage to extract features of Google Landmarks Dataset v2 or INSTRE.

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PWD
python3 tools/compute_candidate_pool.py --cfg configs/instre/base.yaml --task feature --dataset instre --name s0_r50 --pretrain imagenet
```

### Compute Candidate Pool for Each Image
As mentioned in Section 3.1: Setup for training samples.

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PWD
python3 tools/compute_candidate_pool.py --task neighbor --dataset instre --name s0_r50
```

### Configuration
There are two different configs we use in the whole training, we call them `base` and `impr` config, respectively.

For Google Landmarks Dataset v2, We use `base` config for the first three training stages, and `impr` config in the fourth training stage, so the whole training contains four stages. And for INSTRE, only one `base` config training stage is considered. 


### Start Training
```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PWD
tools/train_net.py --cfg /path/to/config
```
