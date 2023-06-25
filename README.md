# Implementation of [Shared Autonomy via Deep Reinforcement Learning](https://www.roboticsproceedings.org/rss14/p05.pdf)

*正在施工 Unfinished

本仓库是对[这篇论文](https://www.roboticsproceedings.org/rss14/p05.pdf)的复现
  
This repo is an **unofficial** implementation of [this paper](https://www.roboticsproceedings.org/rss14/p05.pdf).

This implementation differs from previous work that previous work used tenserflow where this implementation uses PyTorch

Paper info:  
REDDY S, DRAGAN A, LEVINE S. Shared Autonomy via Deep Reinforcement Learning[C/OL]//Robotics: Science and Systems XIV. Robotics: Science and Systems Foundation, 2018. DOI:10.15607/RSS.2018.XIV.005

## 安装 Install

```zsh
conda install -c conda-forge gym
conda install pytorch
```

## 使用 Usage

1. run [pretrain.py](./pretrain.py) to get a pretrained policy network and a target network without human in the loop
2. run [train.py](./train.py) to fine-tuning the networks with human in the loop
3. run [run.py](./run.py) to test the result of the trained networks

## 背景 Background

最近在浅浅地学习一些RL相关内容，建立本repo仅做学习记录。本人工程渣，如有错误，请不吝赐教！  
小弟拜谢。

This repo is for my learning records. My engineering sucks, if there are mistakes, please let me know!  
Thank you bro.

## 参考 References

- [Paper](https://www.roboticsproceedings.org/rss14/p05.pdf)  
- [Deep Q-Learning](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)  
- [Double Q-Learning](https://proceedings.neurips.cc/paper_files/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf)
