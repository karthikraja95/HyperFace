# HyperFace

## Abstract

A multi-task learning convolutional neural network for the purpose of
performing landmark localization and other correlated tasks is studied and
analysed in this project. A different and more challenging task around
landmark localization than the one implemented originally is studied using
a HyperFace architecture. It is seen that having a multi-task CNN helps
improving the accuracy among the correlated tasks. A lot of tasks in
Computer Vision are actually correlated and thus architectures like these
can prove to be very useful. We present extensive results on the training
of multitask network in different scenarios and the complexities in training
them. Though our model is modestly trained on a dataset that is new for
HyperFace, it is found to give reasonably good results.

## Architecture for HyperFace

![HyperFace Architecture](architecture.png)


## Code

The following files are inside the code directory

-  The file exp1and5.py has the implementation of experiments 1 and 5 as mentioned in the report.
   To run this code for different experiments, we need to change the model file for loading weights,
   and change which layers to update and by using which loss function which is just changing 4 lines of code.

- The file exp2(a)_frcnn.py trains the frcnn on Genface as mentioned in the report.

- exp2(b)_frcnn.py does the training as explained in the report for experiment 2(b)
Note that the above two files only train for the frcnn part of the network and not the whole HyperFace.

- exp2(a)_and2(b)_hyperface_train.py trains the hyperface network for corresponding experiments.

- exp3_hyperface_train.py does the training for hyperface as given in experiment 3 in report.

- exp(4)_train.py performs training corresponding to experiment 4 mentiond in the report.

- data_creation_images.ipynb and data_creation_videos.ipynb were used to generate dataset in the form of numpy files used in training.
