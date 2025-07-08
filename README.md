<div align="center">   
  
# ProtoOcc: Accurate, Efficient 3D Occupancy Prediction Using Dual Branch Encoder-Prototype Query Decoder

[**Jungho Kim***](https://scholar.google.com/citations?user=9wVmZ5kAAAAJ&hl=ko), **Changwon Kang***, **Dongyoung Lee***, [**Sehwan Choi**](https://scholar.google.com/citations?user=O2XSTY4AAAAJ&hl=ko&oi=ao), [**Jun Won Choi†**](https://scholar.google.com/citations?user=IHH2PyYAAAAJ&hl=ko&oi=ao)  
<sub>*: Equal Contribution,  †: Corresponding Author</sub>

### **AAAI 2025**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2412.08774)

</div>





## News
- [2025/07]: We released the full code & checkpoints of ProtoOcc, including **nuScenes (Single & Multi frame)** and **SemanticKITTI**.
- [2024/12]: ProtoOcc is accepted at AAAI 2025. 🔥
- [2024/08]: ProtoOcc achieves the SOTA on Occ3D-nuScenes with **45.02% mIoU** (Multi-frame) and **39.56% mIoU, 12.83 FPS** (Single-frame)!
</br>


## ⚡ ProtoOcc Performance
<img src="plot/InferenceTime.png" alt="inference.jpg" width="600">

## Main Result
### nuScenes Result
| Config                              | Temporal | Backbone | Input Size | Pooling Method | mIoU  | Model |
|:----------------------------------:|:-------------:|:--------:|:----------:|:----------:|:-----:|:-----:|
| ProtoOcc_1key                        |   1 Frame    |   R50    |  256x704   |   BEVDepth    | **39.56** |  gdrive     |  
| ProtoOcc_longterm                    |   8 Frames    |   R50    |  256x704   |   BEVStereo    | **45.02** |  gdrive     |  

### Semantic-KITTI Result
| Config                              | Temporal | Backbone | Input Size | Pooling Method | mIoU  | Model |
|:----------------------------------:|:-------------:|:--------:|:----------:|:----------:|:-----:|:-----:|
| ProtoOcc_semanticKITTI               |   1 Frame    |   R50    |  384x1280   |   BEVDepth    | **13.89** |  gdrive    |  

## Get Started
- Environment Setup
- Model Training & Evaluation

## 🙏 Acknowledgement

This project builds upon several outstanding open-source projects. We gratefully acknowledge the following key contributions.

- [open-mmlab](https://github.com/open-mmlab)
- [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [SurroundOcc](https://github.com/weiyithu/SurroundOcc)
- [OccFormer](https://github.com/zhangyp15/OccFormer)
- [FB-OCC](https://github.com/NVlabs/FB-BEV)
- [FlashOCC](https://github.com/Yzichen/FlashOCC)
- [COTR](https://github.com/NotACracker/COTR)

## 📃 Bibtex

If you find this work useful for your research or projects, please consider citing the following BibTeX entry.

```
@inproceedings{kim2025protoocc,
  title={Protoocc: Accurate, efficient 3d occupancy prediction using dual branch encoder-prototype query decoder},
  author={Kim, Jungho and Kang, Changwon and Lee, Dongyoung and Choi, Sehwan and Choi, Jun Won},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={4},
  pages={4284--4292},
  year={2025}
}
```

