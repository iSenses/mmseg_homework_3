#!/usr/bin/env bash

export LD_LIBRARY_PATH=/usr/local/cuda-11.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
pip install -U pip
pip install install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install openmim
echo "openmim here========================================================="
mim install mmengine
mim install mmcv==2.0.0rc4
pip install opencv-python pillow matplotlib seaborn tqdm 'mmdet>=3.0.0rc1'
git clone --recursive https://gitee.com/open-mmlab/mmsegmentation.git -b dev-1.x
cd mmsegmentation
pip install -v -e .
cd ../
echo "done========================================================================================="
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
curl -O https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/dataset/iccv09Data.zip
unzip -q iccv09Data.zip
echo "tree==========================================================="
ls -l
python train.py
