# Layer-Regression
This is the official repository for LR (Layer Regression) as proposed in the paper "Universal and Efficient Detection of Adversarial Data through Nonuniform Impact on Network Layers." In this official implementation, we demonstrate the calculation of detection score proposed in LR. Detection scores are calculated for both clean videos and attacked images. 

Currently, we provide one pre-trained detector and a script to test it on clean and adversarial images. More pre-trained models and a training script for your own models will be added soon. Further details can be found in our [paper](openreview.net/forum?id=0CY5APFnFI).


## Installation

To install the required packages:

```
pip install -r requirements.txt
```

Download pre-trained LR model this [link](https://drive.google.com/file/d/1zSIYrBdqvDlcmQAnAsbli0CaCkoU7mvO) and place under lr-models directory. [Download](https://drive.google.com/file/d/10FQEQ1yOT9Jtd7hESfDbYcT2cqa-Eg5t) and place the validation set of 2012 Imagenet (ILSVRC 2012) under the directory /dataset/imagenet/val.

## Usage

```lr-demo.py``` calculates the LR score for 5 samples both clean and attacked versions, it is expected to have high score for adversarial images and low score for clean images.
