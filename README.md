---
title: CoPS
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: gradio
python_version: 3.10
app_file: app.py
pinned: false
---

# CoPS

![](figures/CoPS_schematic.png)

**CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection**

_Qiyu Chen, Zhen Qu, Wei Luo, Haiming Yao, Yunkang Cao, Yuxin Jiang, Yinan Duan,  
Huiyuan Luo, Chengkan Lv*, Zhengtao Zhang_

~~CVPR DOI Link~~ & 
~~IEEE DOI Link~~ & 
[ArXiv Preprint Link](https://arxiv.org/abs/2508.03447)

## Introduction
This repository provides PyTorch-based source code for CoPS (Accepted by CVPR 2026 Findings),
a zero-shot anomaly detection framework that dynamically synthesizes visually conditioned prompts
to adapt CLIP for more effective anomaly-aware prompt learning.
Here, we present a brief summary of CoPS's performance across 5 industrial and 8 medical datasets.

### 1. Industrial Datasets
| Dataset                                                               | I-AUROC (%) | I-AP (%) | P-AUROC (%) | P-AP (%) |
|-----------------------------------------------------------------------|------------:|---------:|------------:|---------:|
| [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/) |        95.0 |     97.8 |        93.4 |     41.9 |
| [VisA](https://github.com/amazon-science/spot-diff/)                  |        85.4 |     88.0 |        95.7 |     23.4 |
| [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip)              |        93.6 |     94.9 |        94.6 |     42.6 |
| [MPDD](https://github.com/stepanje/MPDD/)                             |        78.6 |     81.1 |        97.5 |     30.9 |
| [DTD-Synthetic](https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1) |        95.2 |     98.1 |        98.4 |     58.5 |

### 2. Medical Datasets
| Dataset          | I-AUROC (%) | I-AP (%) | P-AUROC (%) | P-AP (%) |
|------------------|------------:|---------:|------------:|---------:|
| [HeadCT](https://drive.google.com/file/d/1lSAUkgZXUFwTqyexS8km4ZZ3hW89i5aS/view?usp=sharing) |        96.1 |     97.1 |           – |        – |
| [BrainMRI](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) |        97.4 |     97.6 |           – |        – |
| [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) |        98.7 |     98.5 |           – |        – |
| [ISIC](https://drive.google.com/file/d/1UeuKgF1QYfT1jTlYHjxKB3tRjrFHfFDR/view?usp=sharing) |           – |        – |        93.8 |     85.6 |
| [CVC-ColonDB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579) |           – |        – |        85.6 |     37.2 |
| [CVC-ClinicDB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579) |           – |        – |        88.8 |     49.9 |
| [Kvasir](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579) |           – |        – |        85.8 |     51.5 |
| [Endo](https://drive.google.com/file/d/1LNpLkv5ZlEUzr_RPN5rdOHaqk0SkZa3m/view) |           – |        – |        90.0 |     58.7 |

## Environments
Create a new conda environment and install required packages.
```
conda create -n cops_env python=3.10.19
conda activate cops_env
pip install -r requirements.txt
```
Experiments were conducted on NVIDIA GeForce RTX 3090 (24GB).
Same GPU and package version are recommended. 

## Data Preparation
The download links for the datasets are provided in the introduction table above.
These dataset folders/files follow its original structure.
The `./datasets/json` directory contains the code files used to generate JSON annotations for each industrial and medical dataset. After executing the corresponding file,
a `meta.json` file will be created in the root directory of each dataset.

## Run Experiments

- Testing and visualizing on the unseen categories (use the pre-trained weights)

```
cd ./shell
bash test.sh 0 10 VisA MVTec  # gpu_id=0, epoch=10, trained on VisA,  test on MVTec
bash test.sh 0 10 VisA MPDD   # gpu_id=0, epoch=10, trained on VisA,  test on MPDD
bash test.sh 0 10 VisA HeadCT # gpu_id=0, epoch=10, trained on VisA,  test on HeadCT
......
bash test.sh 0 5 MVTec VisA   # gpu_id=0, epoch=5,  trained on MVTec, test on VisA
```

- Training on the seen categories of auxiliary datasets (train your own weights)

```
cd ./shell
bash train.sh 0 10 VisA; bash test.sh 0 10 VisA MVTec  # train on VisA,  test on MVTec
bash train.sh 0 5 MVTec; bash test.sh 0 5 MVTec VisA   # train on MVTec, test on VisA
```

## Visualization
The visualization results and model files are saved in `./results`.
Some qualitative visualization results on several datasets are shown below.

![](figures/CoPS_visualization.png)

## Citation
Please cite the following paper if the code help your project:

```bibtex
@article{chen2025cops,
  title={Cops: Conditional prompt synthesis for zero-shot anomaly detection},
  author={Chen, Qiyu and Qu, Zhen and Luo, Wei and Yao, Haiming and Cao, Yunkang and Jiang, Yuxin and Duan, Yinan and Luo, Huiyuan and Lv, Chengkan and Zhang, Zhengtao},
  journal={arXiv preprint arXiv:2508.03447},
  year={2025}
}
```

## Acknowledgements
Thanks for the great inspiration from [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP), [AdaCLIP](https://github.com/caoyunkang/AdaCLIP), and [Bayes-PFL](https://github.com/xiaozhen228/Bayes-PFL).

## License
The code in this repository is licensed under the [MIT license](https://github.com/cqylunlun/CoPS?tab=MIT-1-ov-file/).
