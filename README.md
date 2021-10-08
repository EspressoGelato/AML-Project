# AML-Project

## Data Prepartion
  1. Videos should be preprocessed firstly to frames. We use the script from https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/util_scripts/generate_video_jpgs.py to preprocess the MSR-VTT dataset.
  2. UCF101 dataset is preprocessed by 3D-ResNet/util_scripts/generate_video_hdf5.py to genereate corresponding hdf5 files.

## Training
1. To train the virtex-video version, please enter virtex-video, and run 
   ```
    CUDA_VISIBLE_DEVICES=1 python scripts/pretrain_virtex.py --config configs/_base_video_R21D_512_L1_H1024.yaml  --num-gpus-per-machine 1 --cpu-workers 4 --       serialization-dir ./results-2  --checkpoint-every 1000 --dist-url tcp://127.0.0.1:23457
 ```
 2. Our code is heavily based on offical Virtex code, https://github.com/kdexd/virtex
 3. Changes are mainly made in virtex/models/captioning_video.py, virtex/modules/, virtex/data/readers_video.py,spatial_transforms.py,get_mean_std.py temporal_transforms.py , virtex/data/datasets/captioning_video.py, virtex/factories.py. Others are minor changes. 
 
 4. To train 3D ResNet with pretrained weights, please refer our scripts in: 3D-ResNet/scripts/run_baseline_pretrained.sh
 
