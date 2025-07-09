<div align="center">

# 🚀 [DTL Detection Module] Disentangled Transfer Learning for Object Detection & Instance Segmentation

</div>

---

## 🔍 Overview

The **detection/** folder contains the implementation of Disentangled Transfer Learning (DTL) applied to object detection and instance segmentation tasks. It integrates a Compact Side Network (CSN) into popular detection frameworks to achieve parameter and memory efficiency.

---

## 📦 Installation & Setup

### Prepare Environment

Please refer to [MMDetection](https://github.com/open-mmlab/mmdetection) documentation for environment setup and dataset preparation.

If you run into any difficulties, you can use the following commands as a rough guide to set up your environment:
```angular2html
conda create -n dtl python=3.9 -y

# which means your cuda version needs to be 11.1
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install ninja yapf==0.40.1 cython==0.29.33 pycocotools torchsummary
pip install mmpycocotools

# enter detection folder and install mmdetection locally
MMCV_WITH_OPS=1 pip install -e . -v

# apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ --no-build-isolation

#gcc
vim /etc/apt/sources.list
sudo apt-key add armored-keys.asc
sudo apt-get update
sudo apt install gcc-7 g++-7
```

### Prepare Datasets

#### COCO

You can download the COCO 2017 dataset using the following commands:

```bash
mkdir -p data/coco
cd data/coco
# Download images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# Unzip everything
unzip train2017.zip && unzip val2017.zip && unzip test2017.zip
unzip annotations_trainval2017.zip
```

* After extraction, ensure your config’s `data_root` (in `dtl_configs/_base_/coco_instance_sample_1.py`) points to `data/coco`.

#### VOC

```bash
mkdir -p data/VOCdevkit
cd data/VOCdevkit
# VOC2007
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# VOC2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# Extract all archives
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
tar -xvf VOCtrainval_11-May-2012.tar
```

* After extraction, ensure your config’s `data_root` (in `dtl_configs/_base_/voc0712.py`) points to `data/VOCdevkit`.

### Pretrained Models

| name | pretrain | resolution | #params | FLOPs | FPS| 22K model |
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-B | ImageNet-22K | 224x224 |  88M | 15.4G | 278 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth)/[baidu](https://pan.baidu.com/s/1y1Ec3UlrKSI8IMtEs-oBXA)/[config](configs/swin/swin_base_patch4_window7_224_22k.yaml) | 
| Swin-L | ImageNet-22K | 224x224 | 197M | 34.5G | 141 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth)/[baidu](https://pan.baidu.com/s/1pws3rOTFuOebBYP3h6Kx8w)/[config](configs/swin/swin_large_patch4_window7_224_22k.yaml) | 

* Set `load_from` to the path of your pre-trained weights.


## 🛠️ Training DTL


#### COCO

```bash
bash tools/dist_train.sh dtl_configs/swin-b_coco/cascade_mask_swin_base_3x_coco_sample_1_bs_16_dtlv2_ls.py YOUR_GPUS
```

#### VOC

```bash
bash bash tools/dist_train.sh dtl_configs/swin-l_voc/voc_retinanet_swin_large_1x_dtlv2_ls.py YOUR_GPUS
```

## 🙏 Acknowledgements
We gratefully acknowledge the following outstanding open-source projects and libraries, among others:

- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [NOAH](https://github.com/ZhangYuanhan-AI/NOAH)
- [LoRA](https://github.com/microsoft/LoRA)
- [Adaptformer](https://github.com/ShoufaChen/AdaptFormer)
- [SSF](https://github.com/dongzelian/SSF)
- [VPT](https://github.com/KMnP/vpt) 
- [timm](https://github.com/rwightman/pytorch-image-models)
- **Special thanks to [MONA](https://github.com/LeiyiHU/mona)**, on which this detection codebase is built.
