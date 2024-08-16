# DarkHash

The implementation of our paper "DarkHash: A Data-Free Backdoor Attack Against Deep Hashing".

## Setup Environment

Before you begin, make sure you have installed the following dependencies:

- Python 3.8
- PyTorch 1.12.1
- TorchVison 0.13.1

You can install and set up the project by following these steps:

## Data Preparation

Download the ImageNet dataset from the [ImageNet](http://www.image-net.org/download-images) and download the Pascal VOC dataset from the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html), the path is as follows:

```
DarkHash
├── Dataset
│   ├── ImageNet
│   │   ├── ILSVRC2012
│   │   │   ├── ILSVRC2012_val_00000001.JPEG
│   │   │   ├── ······
│   │   ├── n01514668
│   │   ├── ······
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
```

Then you can obtain the data information from [Link](https://github.com/swuxyj/DeepHash-pytorch/tree/master/data).

## Prepare

Run shadow_dataset.py to generate shadow dataset:

```
cd DarkHash
python shadow_dataset.py --shadow_data ImageNet --victim_data voc2012 --bit 64 --backbone ResNet50 --hash_method HashNet
```

The shadow dataset will be saved to `./Dataset/ResNet50/HashNet_ImageNet_voc2012_64bit_dataset_feature`.

## Attack 

Run attack.py to train the backdoored model:

```
python attack.py --shadow_data ImageNet --model_dataset voc2012 --bit 64 --backbone ResNet50 --hash_method HashNet
```

