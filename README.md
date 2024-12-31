# Graph Convolution Network for fMRI Analysis Based on Connectivity Neighborhood
Here is the [paper](https://www.mitpressjournals.org/doi/pdf/10.1162/netn_a_00171). Please feel free to send [email](mailto:lebo.wang@email.ucr.edu) for any question.


## Overview
We propose a connectivity-based graph convolution network (`cGCN`) architecture for fMRI analysis. fMRI data are represented as the k-nearest neighbors graph based on the group functional connectivity, and spatial features are extracted from connectomic neighborhoods through Graph Convolution Networks. We have demonstrated our cGCN architecture on two scenarios with improved classification accuracy (individual identification on HCP dataset and classification of patients from normal controls on ABIDE dataset). cGCN on the graph-represented data can be extended to fMRI data in other data representations, which provides a promising deep learning architecture for fMRI analysis.

The architecture of cGCN is shown below:

![Architecture of cGCN](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/fig/Figure1.JPG)

## How to run/test
### HCP dataset

Easily try out on [colab](https://colab.research.google.com/drive/1qTRZaAa4FO4xckKK_dKRuFfVhPZGSzIS?usp=sharing). Or run the [notebook](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/demo.ipynb) locally. 

Alternatively, set up the environment (refer to [dependencies](#dependencies)) and download data:

- [data](https://drive.google.com/drive/folders/1akwp8DVEUEMD-WA2aNbF0uSbmh_NOFvl?usp=sharing)
- ~[Data](https://drive.google.com/file/d/1l029ZuOIUY5gehBZCAyHaJqMNuxRHTFc/view?usp=sharing) (Individual identification on 100 unrelated subjects)~
- ~[FC matrix](https://drive.google.com/file/d/1WP4_9bps-NbX6GNBnhFu8itV3y1jriJL/view?usp=sharing)~
- ~[Model](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/model.py)~
- ~[Pre-trained model](https://drive.google.com/file/d/1KePCfQOt1hk6TfL98Y4qnsFvdTSYPijh/view?usp=sharing)~

Then you can run [run_HCP.py](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/run_HCP.py).

### ABIDE dataset

Please set up the environment (refer to [dependencies](#dependencies)) and download data:

- [data](https://drive.google.com/drive/folders/1akwp8DVEUEMD-WA2aNbF0uSbmh_NOFvl?usp=sharing)
- ~[Leave-one-site-out dataset](https://drive.google.com/file/d/1xer4TMU1fqbwDO2wBOGrnIFuQ4v4Y3-o/view?usp=sharing)~
- ~[10-fold dataset](https://drive.google.com/file/d/1RhMRzDRT2vAkXDiW4t55Wbt8XRi6f9_x/view?usp=sharing)~
- ~[Model](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/ABIDE/model.py)~

Then you can run [run_ABIDE_leave_one_site_out.py](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/ABIDE/run_ABIDE_leave_one_site_out.py) and [run_ABIDE_10_fold.py](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/ABIDE/run_ABIDE_10_fold.py).


## Results

### HCP dataset

cGCN is tested and compared with ConvRNN [Wang, Lebo, et al. "Application of convolutional recurrent neural network for individual recognition based on resting state fmri data." Frontiers in Neuroscience 13 (2019): 434.]. The relation between the accuracy and the number of input frames (the random chance is 1%) is shown below:

![HCP](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/fig/Figure2.JPG)

### ABIDE dataset

cGCN is tested and compared with DNN [Heinsfeld, Anibal Sólon, et al. "Identification of autism spectrum disorder using deep learning and the ABIDE dataset." NeuroImage: Clinical 17 (2018): 16-23.]. Both the leave-one-site-out and 10-fold cross-validations were tested and compared as shown below:

![ABIDE 1](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/fig/Figure3.jpg)

For the leave-one-site-out cross-validation, the relation between the accuracy and the number of input frames is shown below:

![ABIDE 2](https://github.com/Lebo-Wang/cGCN_fMRI/blob/master/fig/Figure4.jpg)

## Dependencies

- keras=2.1.5
- tensorflow=1.4.1
- h5py=2.8.0
- nilearn=0.5.0
- numpy=1.15.4

## Acknowledgement
Some code is borrowed from [dgcnn](
https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/utils/tf_util.py).
