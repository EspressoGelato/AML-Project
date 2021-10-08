#!/usr/bin/env bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

ROOT=$PWD
FOLDER=ucf51_pretrained_prob
MODEL=r21d
DEPTH=18
VIDEO_PATH=datasets/UCF101/hdf5data
AUDIO_PATH=datasets/UCF101/ucf101_audiocnn14embed512_features
Pretrained_PATH=pretrained/results-3/checkpoint_35000.pth

mkdir results/${FOLDER}

### testing recognition
python main.py \
--root_path ${ROOT} \
--video_path ${VIDEO_PATH} \
--annotation_path datasets/UCF101/ucf51_01.json \
--result_path results/${FOLDER} \
--resume_path results/${FOLDER}/save_model.pth \
--dataset ucf101 --n_classes 51 \
--model ${MODEL} --model_depth ${DEPTH}  \
--file_type hdf5 --sample_t_stride 1 \
--n_threads 16 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1

### print acc
python -m util_scripts.eval_accuracy datasets/UCF101/ucf51_01.json results/${FOLDER}/val.json --subset validation -k 1 --ignore
