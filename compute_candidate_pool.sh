export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD
python3 tools/compute_candidate_pool.py --cfg configs/instre/base.yaml --task feature --dataset instre --name s0_r50 --pretrain imagenet
# python3 tools/compute_candidate_pool.py --task neighbor --dataset instre --name s0_r50
