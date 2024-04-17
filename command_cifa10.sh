#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

commands=(
  "python /root/code/MLKD/train.py --cfg /root/code/MLKD/configs/cifar10/dkd/our_MLKD_DKD_vgg16_r20.yaml"
  "python /root/code/MLKD/train.py --cfg /root/code/MLKD/configs/cifar10/dkd/our_MLKD_DKD_vgg16_r32.yaml"
  "python /root/code/MLKD/train.py --cfg /root/code/MLKD/configs/cifar10/dkd/our_MLKD_DKD_vgg16_r44.yaml"
  "python /root/code/MLKD/train.py --cfg /root/code/MLKD/configs/cifar10/dkd/our_MLKD_DKD_res56_r20.yaml"
  "python /root/code/MLKD/train.py --cfg /root/code/MLKD/configs/cifar10/dkd/our_MLKD_DKD_res56_r32.yaml"
  "python /root/code/MLKD/train.py --cfg /root/code/MLKD/configs/cifar10/dkd/our_MLKD_DKD_res56_r44.yaml"
  )

# 定义运行次数
num_runs=10

# 循环执行命令
for ((i=0; i<$num_runs; i++)); do
  for command in "${commands[@]}"; do
    $command
  done
done