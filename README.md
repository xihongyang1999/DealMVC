[stars-img]: https://img.shields.io/github/stars/xihongyang1999/ICL_SSL?color=yellow
[stars-url]: https://github.com/xihongyang1999/ICL_SSL/stargazers
[fork-img]: https://img.shields.io/github/forks/xihongyang1999/ICL_SSL?color=lightblue&label=fork
[fork-url]: https://github.com/xihongyang1999/ICL_SSL/network/members
[visitors-img]: https://visitor-badge.glitch.me/badge?page_id=xihongyang.1999.ICL_SSL/
[adgc-url]: https://github.com/xihongyang1999/ICL_SSL

# Interpolation-based Contrastive Learning for Few-Label Semi-Supervised Learning

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://cis.ieee.org/publications/t-neural-networks-and-learning-systems" alt="Journal">
        <img src="https://img.shields.io/badge/IEEE TNNLS'22-brightgreen" /></a>
<p/>



[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
[![visitors][visitors-img]][adgc-url]


An official source code for paper Interpolation-based Contrastive Learning for Few-Label Semi-Supervised Learning, accepted by IEEE TNNLS 2022. Any communications or issues are welcomed. Please contact xihong_edu@163.com. If you find this repository useful to your research or work, it is really appreciate to star this repository. :heart:

-------------

### Overview

<p align = "justify"> 
 Illustration of Interpolation-based Contrastive Learning Semi-Supervised Learning (ICL-SSL) mechanism. 
</p>
<div  align="center">    
    <img src="./assets/overall.png" width=60%/>
</div>







### Requirements

The proposed ICL_SSL is implemented with python 3.8.8 on a NVIDIA 1080 Ti GPU. 

Python package information is summarized in **requirements.txt**:

- torch==1.8.0
- tqdm==4.61.2
- numpy==1.21.0
- tensorboard==2.8.0



### Quick Start

```
python train.py 
```



### Citation

If you use code or datasets in this repository for your research, please cite our paper.

```
@ARTICLE{ICL_SSL,
  author={Yang, Xihong and Hu, Xiaochang and Zhou, Sihang and Liu, Xinwang and Zhu, En},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Interpolation-Based Contrastive Learning for Few-Label Semi-Supervised Learning}, 
  year={2022},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TNNLS.2022.3186512}}

```

