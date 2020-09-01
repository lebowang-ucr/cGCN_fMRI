# Graph Convolution Network for fMRI Analysis Based on Connectivity Neighborhood

Please try out the demo: https://colab.research.google.com/drive/1qTRZaAa4FO4xckKK_dKRuFfVhPZGSzIS?usp=sharing. Or run the [notebook](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/demo.ipynb) locally.

## Overview
We propose a connectivity-based graph convolution network (`cGCN`) architecture for fMRI analysis. fMRI data are represented as the k-nearest neighbors graph based on the group functional connectivity, and spatial features are extracted from connectomic neighborhoods through Graph Convolution Networks. We have demonstrated our cGCN architecture on two scenarios with improved classification accuracy (individual identification on HCP dataset and classification of patients from normal controls on ABIDE dataset). cGCN on the graph-represented data can be extended to fMRI data in other data representations, which provides a promising deep learning architecture for fMRI analysis.

The architecture of cGCN is shown below:

![Architecture of cGCN](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/fig/Figure1.JPG)

## Results

### HCP dataset

cGCN is tested and compared with ConvRNN [Wang, L., Li, K., Chen, X., & Hu, X. P. (2019). Application of Convolutional Recurrent Neural Network for Individual Recognition Based on Resting State fMRI Data. Frontiers in Neuroscience.]. The relation between the accuracy and the number of input frames (the random chance is 1%) is shown below:

![HCP](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/fig/Figure2.JPG)

### ABIDE dataset

cGCN is tested and compared with DNN [Heinsfeld, A. S., Franco, A. R., Craddock, R. C., Buchweitz, A., & Meneguzzi, F. (2018). Identification of autism spectrum disorder using deep learning and the ABIDE dataset. NeuroImage: Clinical, 17, 16-23. ]. Both the leave-one-site-out and 10-fold cross-validations were tested and compared as shown below:

![ABIDE 1](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/fig/Figure3.JPG)

For the leave-one-site-out cross-validation, the relation between the accuracy and the number of input frames is shown below:

![ABIDE 2](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/fig/Figure4.JPG)

## Dependencies

- keras==2.1.5
- tensorflow==1.4.1
- h5py==2.8.0
- nilearn==0.5.0
- numpy==1.15.4

## Acknowledgement
Some code is borrowed from [dgcnn](
https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/utils/tf_util.py).
