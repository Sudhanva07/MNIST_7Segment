# MNIST → Seven-Segment Digit Classifier

An experiment in encoding MNIST digit labels as 7-segment display patterns and
training a deep learning pipeline to predict them directly from handwritten
digit images.

## Overview

Instead of the usual 10-way one-hot classification, each digit label (0–9) is
re-encoded as a 7-bit vector representing which segments of a classic
seven-segment display would be lit for that digit (e.g. `0 → [1,1,1,1,1,1,0]`).
The model then learns to predict this 7-bit pattern as a **multi-label binary
classification** problem rather than a standard multi-class one.

The pipeline has two stages:

1. **Autoencoder** – A convolutional autoencoder is trained on the raw MNIST
   images (unsupervised) to learn a denoised/reconstructed version of each
   digit image.
2. **Multi-label CNN classifier** – A second CNN takes the *autoencoder's
   reconstructed images* as input and predicts the 7-segment bit vector using
   a sigmoid output layer + binary cross-entropy loss.

Predictions are thresholded at 0.5, decoded back from 7-segment patterns to
digit labels (0–9), and evaluated with standard classification metrics and a
confusion matrix.

## Pipeline

MNIST images → Autoencoder (reconstruction) → CNN (multi-label, 7 sigmoid outputs)

↓

7-bit segment vector → decoded digit (0-9)

## Results

- Autoencoder: trained 50 epochs, binary cross-entropy reconstruction loss
  converging to ~0.108 (train) / ~0.108 (val).
- Classifier: trained 10 epochs on the reconstructed images.
- Final test performance (micro-averaged): **~96.5% accuracy / precision /
  recall / F1**.

## Requirements

- `tensorflow` / `keras`
- `tensorflow_addons` *(imported but unused — see Possible Improvements)*
- `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

## Usage

Open `MNIST_7Seg.ipynb` and run all cells. The notebook downloads MNIST via
`keras.datasets.mnist`, so no manual data download is needed.
