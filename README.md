# Entangled Watermarks as a Defense against Model Extraction

This repository is an implementation of the paper [Entangled Watermarks as a Defense against Model Extraction](https://arxiv.org/abs/2002.12200). In this repository, we show how to train a watermarked DNN model that is robust against model extraction. The high-level idea is that a special watermark is designed such that it could be used to verify the owenrship of the model if it is stolen by model extraction. For more details, please read the paper.

We test our code on five datasets: MNIST, Fashion-MNIST, Google Speech Commands (10-classes), CIFAR-10, and CIFAR-100.

### Dependency
Our code is implemented and tested on Tensorflow. Following packages are used by the training code.
```
tensorflow==1.14.0
```
Preprocessing code for Google Speech Commands is modifed based on [this github repository](https://github.com/douglas125/SpeechCmdRecognition). The required packages include:
```
keras==2.2.5
kapre==0.1.3.1
librosa==0.6
tqdm
```
To run the codes on CIFAR-10 or CIFAR-100, `tensorflow_datasets` is required.

SVHN is used as out-of-distribution (OOD) watermark (optional, definition can be found in the paper) for CIFAR-10 and CIFAR-100, `scipy` is needed to load SVHN.

### Preprocessing and Training
[MNIST](http://yann.lecun.com/exdb/mnist/) and [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) need to be downloaded (4 `.gz` files as provided on the corresponding website) into the `data` folder. Then use the following line to conver them to `.pkl` file. 
```
python prepare_mnist.py --dataset {mnist/fashion}
```
Note that to use OOD watermark for MNIST or Fashion MNIST, both datasets need to be saved as `.pkl` files in the `data` folder.

For Google Speech Command dataset, downloading is included in the preprocssing script:
```
python prepare_speechcmd.py
```
To use OOD watermark, add the flag `--OOD [one to nine]` to the line above.

For CIFAR datasets, no preprocessing script is needed. But to use OOD watermark, `train_32x32.mat` from [SVHN](http://ufldl.stanford.edu/housenumbers/) needs to be downloaded to the `data` folder.

After preprocessing, a watermarked DNN model could be trained by the following line. 
```
python train.py --dataset [mnist/fashion/speechcmd/cifar10/cifar100] --default 1
```
There are a number of arguments that could be used to set the hyperparameters. The interpretation and configuration of these hyperparameters are explained in our [paper](https://arxiv.org/abs/2002.12200). Note that by setting the flag `--default 1`, pre-defined hyperparameters will be used.
The `train.py` script also contains a model extraction attack to test the robustness of the watermarks. It is only for testing purpose and is not necessary for training the model.

### Questions or suggestions
If you have any questions or suggestions, feel free to send me an email at nickhengrui.jia@mail.utoronto.ca
