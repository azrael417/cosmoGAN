#!/bin/bash

#some exports
export SLURM_NNODES=1
export SLURM_PROCID=0
export CUDA_VISIBLE_DEVICES=1

#add this to library path:
export PYTHONPATH=$(pwd)/../networks

#clean cp
rm -r checkpoints/*

#run training
python -u ../networks/run_ot_gan.py
