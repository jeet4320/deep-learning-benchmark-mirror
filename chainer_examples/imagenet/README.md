# Large Scale ConvNets

## Modifications
train_imagenet_data_parallel.py script has been modified to run for synthetic data and changes are enclosed within ### Sythetic block
Only resnet50 model is supported for synthetic data. Note the extra parameters added: --dataset and --samples

```
python train_imagenet_data_parallel.py --arch resnet50 --dataset synthetic --epoch 30 --loaderjob 10 --samples 10000 --batchsize 128 --test ~/ ~/ -g 0 1 2 3 4 5 6 7
```

## Requirements

- Pillow (Pillow requires an external library that corresponds to the image format)

## Description

This is an experimental example of learning from the ILSVRC2012 classification dataset.
It requires the training and validation dataset of following format:

* Each line contains one training example.
* Each line consists of two elements separated by space(s).
* The first element is a path to 256x256 RGB image.
* The second element is its ground truth label from 0 to 999.

The text format is equivalent to what Caffe uses for ImageDataLayer.
This example currently does not include dataset preparation script.

This example requires "mean file" which is computed by `compute_mean.py`.
