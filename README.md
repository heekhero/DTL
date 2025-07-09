<div align=center>
  <img src="imgs/DTL_illustration.png" width="500px" />
</div>

# Disentangled Transfer Learning (DTL)

Implementation of **DTL**, a parameter-efficient paradigm that disentangles trainable parameters from a backbone model via a Compact Side Network (CSN). This repository supports both image classification and object detection tasks.

---


## 🚀 Features

- **Memory-efficient**: significantly reduces GPU memory footprint by offloading trainable parameters to CSN.  
- **Parameter-efficient**: only a small fraction of parameters (LoRA-style) is fine-tuned.  
- **Accurate**: achieves state-of-the-art performance on ImageNet, VTAB, and COCO benchmarks.  
- **Versatile**: easy extension to image classification and object detection frameworks.  
- **Seamless**: integrates with PyTorch and Detectron2 pipelines.


---

## 🛠️ Usage

- **Classification:** see `classification/README.md` for training and evaluation steps.  
- **Detection:** see `detection/README.md` for COCO training and evaluation instructions.

