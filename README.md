# DynaPrompt: Dynamic Test-Time Prompt Tuning

This repository provides the official PyTorch implementation of our ICLR 2025 paper:    

> DynaPrompt: Dynamic Test-Time Prompt Tuning
> 
> Zehao Xiao, Shilin Yan, Jack Hong, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiayi Shen, Qi Wang, Cees G. M. Snoek

For more dtails, please check out our [<ins>**paper**</ins>](https://arxiv.org/abs/2501.16404). 

## Overview
This repository contains the implementation of DynaPrompt for image classification with a pre-trained CLIP, 
focusing on online and dynamically adapt the learnable prompts at test time.

## Prerequisites

### Hardware

This implementation is for the single-GPU configuration, evaluated on a single A6000. 

### Environments 
The code is tested on PyTorch 1.13.1. 

The code is based on [CoOp](https://github.com/KaiyangZhou/CoOp) and [TPT](https://github.com/azshue/TPT/tree/main), with similar required packages to them, such as the [dassl](https://github.com/KaiyangZhou/Dassl.pytorch).

### Datasets 

For out-of-distribution generalization, we consider 5 datasets:

* [ImageNet](https://image-net.org/index.php) 
* [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
* [ImageNet-R](https://github.com/hendrycks/imagenet-r)
* [ImageNet-V2](https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz)
* [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

For cross-datasets generalization, we consider 10 datasets:
* [Flower102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
* [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)
* [OxfordPets](https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz)
* [StanfordCars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* [UCF101](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing)
* [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz)
* [Food101](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)
* [SUN397](http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz)
* [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)
* [EuroSAT](http://madm.dfki.de/files/sentinel/EuroSAT.zip)

For cross-dataset generalization, we adopt the same train/val/test splits as CoOp. Please refer to [this page](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#how-to-install-datasets), and look for download links of `split_zhou_${dataset_name}.json`, and put the json files under `./data/data_splits/`. 


## Run DynaPrompt

We provide the bash scripts under `./scripts`. You can directly run bash scripts `run_dynap.sh` to run the codes with the default hyperparameters.     

An example to run DynaPrompt on out-of-distribution datasets:
```
bash run_dynap.sh
```

Change the `data_root` to your own data path.

Change the `dataset` to `A`, `R`, `V`, `S`, or `I` to evaluate on datasets `ImageNet-A`, `ImageNet-R`, `ImageNet-V2`, `ImageNet-Sketch`, or `ImageNet`, respectively.
You can also change the dataset to `flower102`, `dtd`, `pets`, `cars`, `ucf101`, `caltech101`, `food101`, `sun397`, `aircraft`, or `eurosat` for cross-dataset generalization.

Change the `num_p` for different numbers of prompts

Change the `arch` to `RN50` or `ViT-B/16` for different backbones.

`log_date`, `lr`, `ntx`, and `seed` denote the name of log files, learning rate, length of prompts, and random seed, respectively.


## Citation
If you find our code useful or our work relevant, please consider citing: 
```
@inproceedings{
zehao2025dynaprompt,
title={DynaPrompt: Dynamic Test-Time Prompt Tuning},
author={Xiao, Zehao and Yan, Shilin and Hong, Jack and Cai, Jiayin and Jiang, Xiaolong and Hu, Yao and Shen, Jiayi and Wang, Qi and Snoek, Cees GM},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
}
```

## Acknowledgements
We thank the authors of [TPT](https://github.com/azshue/TPT/tree/main) and [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp) for their open-source implementation and instructions on data preparation. 
