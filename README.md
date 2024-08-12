# Toy_HF

## Introduction

Are you puzzled by the fact that there are so many GPUs, yet most of them seem to be in use? :joy:

Do you want to utilize a GPU in a cluster or VM but find that none are available? :blush:

This repository can help you! :rocket:

## How to use it

```bash
conda create -n toyhf python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirments.txt
git clone https://github.com/Zanghu-ze/Toy_HF
cd Toy_HF
python inference.py
```

