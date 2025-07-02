# Layer-Regression

This is the official repository for LR (Layer Regression), as proposed in the paper *"Universal and Efficient Detection of Adversarial Data through Nonuniform Impact on Network Layers."* In this implementation, we demonstrate how to compute the detection scores introduced in LR. These scores are calculated for both clean and adversarial images.

Currently, we provide one pre-trained detector along with a script to test it on clean and adversarial images. Additional pre-trained models and a training script for custom models will be added soon. Further details can be found in our [paper](https://openreview.net/forum?id=0CY5APFnFI).



## Installation

To install the required packages:

```
pip install -r requirements.txt
```

Download the pre-trained LR model from [this link](https://drive.google.com/file/d/1zSIYrBdqvDlcmQAnAsbli0CaCkoU7mvO) and place it in the `lr-models` directory.  
[Download](https://drive.google.com/file/d/10FQEQ1yOT9Jtd7hESfDbYcT2cqa-Eg5t) and place the validation set of the 2012 ImageNet (ILSVRC 2012) in the directory `/dataset/imagenet/val`.

## Usage

Run `lr-demo.py` to calculate the LR scores for 5 samples, including both clean and adversarial versions. The LR score is expected to be high for adversarial images and low for clean images.

