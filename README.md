# Entangled Watermarks as a Defense against Model Extraction

This repository is an implementation of the paper [Entangled Watermarks as a Defense against Model Extraction](https://PLACEHOLDER). In this repository, we show how to train a watermarked DNN model that is robust against model extraction. The high-level idea is that a special watermark is designed such that it could be used to verify the owenrship of the model if it is stolen by model extraction. For more details, please read the paper.

We test our code on three datasets: MNIST, Fashion-MNIST, and Google Speech Commands (10-classes). The code for the two vision datasets (MNIST and Fashion-MNIST) and the audio dataset (Google Speech Commands) are in `\vision` and `\audio` respectively.

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

### Preprocessing and Training
For the vision datasets, [MNIST](http://yann.lecun.com/exdb/mnist/) and [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) need to be downloaded as `.gz` files into the same folder as the scripts. For Google Speech Command dataset, downloading is included in the `preprocess.py`.
To preprocess the datasets, the following line needs to be run.
```
python preprocess.py
```
After preprocessing, a watermarked DNN model could be trained by the following line. 
```
python main.py
```
There are a number of arguments that could be used to set the hyperparameters. The interpretation and configuration of these hyperparameters are explained in our [paper](https://PLACEHOLDER).

### Questions or suggestions
If you have any questions or suggestions, feel free to send me an email at nickhengrui.jia@mail.utoronto.ca
