# Domain-adaptive deep network compression

The paper will appear in ICCV 2017. An [arXiv pre-print](https://arxiv.org/abs/1709.01041) is available.

ICCV 2017 [open access](http://openaccess.thecvf.com/content_ICCV_2017/papers/Masana_Domain-Adaptive_Deep_Network_ICCV_2017_paper.pdf) is available and the poster can be found [here](./pdf/poster_DALR_ICCV_2017.pdf).

## Citation

'''
@InProceedings{Masana_2017_ICCV,
author = {Masana, Marc and van de Weijer, Joost and Herranz, Luis and Bagdanov, Andrew D. and Alvarez, Jose M.},
title = {Domain-Adaptive Deep Network Compression},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
'''

## Authors

Marc Masana, Joost van de Weijer, Luis Herranz, Andrew D. Bagdanov and Jose M. Álvarez

## Institutions

[Computer Vision Center, Barcelona, Spain](http://www.cvc.uab.es/lamp/)

Media Integration and Communication Center, University of Florence, Florence, Italy

Toyota Research Institute

## Abstract

Deep Neural Networks trained on large datasets can be easily transferred to new domains with far fewer labeled examples by a process called fine-tuning. This has the advantage that representations learned in the large source domain can be exploited on smaller target domains. However, networks designed to be optimal for the source task are often prohibitively large for the target task. In this work we address the compression of networks after domain transfer. 

We focus on compression algorithms based on low-rank matrix decomposition. Existing methods base compression solely on learned network weights and ignore the statistics of network activations. We show that domain transfer leads to large shifts in network activations and that it is desirable to take this into account when compressing. We demonstrate that considering activation statistics when compressing weights leads to a rank-constrained regression problem with a closed-form solution. Because our method takes into account the target domain, it can more optimally remove the redundancy in the weights. Experiments show that our Domain Adaptive Low Rank (DALR) method significantly outperforms existing low-rank compression techniques. With our approach, the fc6 layer of VGG19 can be compressed more than 4x more than using truncated SVD alone -- with only a minor or no loss in accuracy. When applied to domain-transferred networks it allows for compression down to only 5-20% of the original number of parameters with only a minor drop in performance.
