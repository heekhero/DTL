# Disentangled Transfer Learning (DTL)

Implementation of **DTL**, a parameter-efficient paradigm that disentangles trainable parameters from a backbone model via a Compact Side Network (CSN). This repository supports both image classification and object detection tasks.

---

## 📖 Paper

> **Minghao Fu**, Ke Zhu, Jianxin Wu  
> **DTL: Disentangled Transfer Learning for Visual Recognition**, [arxiv](https://arxiv.org/abs/2312.07856) • [AAAI Paper](https://dl.acm.org/doi/10.1609/aaai.v38i11.29096)

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

## 🎓 Citation

If you use DTL in your work, please cite:

```bibtex
@inproceedings{10.1609/aaai.v38i11.29096,
author = {Fu, Minghao and Zhu, Ke and Wu, Jianxin},
title = {{DTL}: disentangled transfer learning for visual recognition},
year = {2024},
booktitle = {Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence},
pages={12082 - 12090}
}
```