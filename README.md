# Domain-adaptive deep network compression

ICCV 2017 [open access](http://openaccess.thecvf.com/content_ICCV_2017/papers/Masana_Domain-Adaptive_Deep_Network_ICCV_2017_paper.pdf) is available and the poster can be found [here](./pdf/poster_DALR_ICCV_2017.pdf). The [arXiv pre-print](https://arxiv.org/abs/1709.01041) is also available.

## How to run the tensorflow code

The example is done on a vanilla non-domain-transfer simple experiment. We train a LeNet network from scratch on MNIST dataset and then compress the network using either the SVD baseline or our proposed DALR method. The example code is given in a jupyter notebook.
```
cd code/tensorflow
jupyter notebook Experiment_LeNet_MNIST.ipynb
```

## How to run the matlab code

The example network can be downloaded from [here](http://mmasana.foracoffee.org/DALR_ICCV_2017/birds_vgg19_net.mat) and copied to a new folder "nets/".
```
mkdir nets
cd nets
wget http://mmasana.foracoffee.org/DALR_ICCV_2017/birds_vgg19_net.mat
```
Then, the example can be run from the "code/" folder by calling the "mainScript_compress_DALR.m" file on the MatLab terminal.

## Citation

```
@InProceedings{Masana_2017_ICCV,
author = {Masana, Marc and van de Weijer, Joost and Herranz, Luis and Bagdanov, Andrew D. and Alvarez, Jose M.},
title = {Domain-Adaptive Deep Network Compression},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
```
Code by [Marc Masana](https://mmasana.github.io/), PhD student at
[LAMP research group](http://www.cvc.uab.es/lamp/) at Computer Vision Center, Barcelona


## Abstract

Deep Neural Networks trained on large datasets can be easily transferred to new domains with far fewer labeled examples by a process called fine-tuning. This has the advantage that representations learned in the large source domain can be exploited on smaller target domains. However, networks designed to be optimal for the source task are often prohibitively large for the target task. In this work we address the compression of networks after domain transfer. 

We focus on compression algorithms based on low-rank matrix decomposition. Existing methods base compression solely on learned network weights and ignore the statistics of network activations. We show that domain transfer leads to large shifts in network activations and that it is desirable to take this into account when compressing. We demonstrate that considering activation statistics when compressing weights leads to a rank-constrained regression problem with a closed-form solution. Because our method takes into account the target domain, it can more optimally remove the redundancy in the weights. Experiments show that our Domain Adaptive Low Rank (DALR) method significantly outperforms existing low-rank compression techniques. With our approach, the fc6 layer of VGG19 can be compressed more than 4x more than using truncated SVD alone -- with only a minor or no loss in accuracy. When applied to domain-transferred networks it allows for compression down to only 5-20% of the original number of parameters with only a minor drop in performance.
