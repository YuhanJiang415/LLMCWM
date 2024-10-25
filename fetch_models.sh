#!/bin/bash

MODELS_URL="https://zenodo.org/records/13992827/files/pretrained_models.zip?download=1"
TARGET_DIR="pretrained_models/"

mkdir -p $TARGET_DIR

echo "Downloading pretrained models..."
wget -O pretrained_models.zip $MODELS_URL

echo "Extracting models..."
unzip pretrained_models.zip -d $TARGET_DIR

rm pretrained_models.zip

echo "Pretrained models have been downloaded and extracted to $TARGET_DIR."
