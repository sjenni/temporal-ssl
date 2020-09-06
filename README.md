## Video Representation Learning by Recognizing Temporal Transformations [[Project Page]](https://sjenni.github.io/temporal-ssl/) 

[Simon Jenni](https://sjenni.github.io), [Givi Meishvili](https://gmeishvili.github.io), and [Paolo Favaro](http://www.cvg.unibe.ch/people/favaro).  
In [ECCV](https://arxiv.org/abs/2007.10730), 2020.

![Model](https://sjenni.github.io/temporal-ssl/assets/time_warps.png)


This repository contains code for self-supervised pre-training on UCF101 and supervised transfer learning on the UCF101 and HMDB51 action recognition benchmarks.

## Requirements
The code is based on Python 3.7 and tensorflow 1.15. 

## How to use it

### 1. Setup

- Dowload [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) and [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).
- Set the paths to the data and log directories in [constants.py](constants.py).
- Run [init_datasets.py](init_datasets.py) to  convert the UCF101 and HMDB datasets to the TFRecord format:
```
python init_datasets.py
```

### 2. Training and evaluation 

- To train and evaluate a model using the C3D architecture, execute [train_test_C3D.py](train_test_C3D.py). An example usage could look like this: 
```
python train_test_C3D.py --tag='test' --num_gpus=2
```

