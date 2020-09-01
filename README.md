# Graph Convolution Network for fMRI Analysis Based on Connectivity Neighborhood

Please try out the demo: https://colab.research.google.com/drive/1qTRZaAa4FO4xckKK_dKRuFfVhPZGSzIS?usp=sharing. Or run the [notebook](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/demo.ipynb) locally.

## Overview
we propose a connectivity-based graph convolution network (`cGCN`) architecture for fMRI analysis. fMRI data are represented as the k-nearest neighbors graph based on the group functional connectivity, and spatial features are extracted from connectomic neighborhoods through Graph Convolution Networks (GCNs). We have demonstrated our cGCN architecture on two scenarios with improved classification accuracy (individual identification on HCP dataset and classification on ABIDE dataset). GCNs on the graph-represented data can be extended to fMRI data in other data representations, which provides a promising deep learning architecture for fMRI analysis.

## Dependencies

- keras==2.1.5
- tensorflow==1.4.1
- h5py==2.8.0
- nilearn==0.5.0
- numpy==1.15.4

## Acknowledgment
Some code is borrowed from [dgcnn](
https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/utils/tf_util.py).
