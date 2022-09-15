# SimpleViT
Simple implementation of Vision Transformer for Image Classification.

- DRL framework : PyTorch

## Install
```bash
git clone https://github.com/isk03276/SimpleViT
cd SimpleViT
pip install -r requirements.txt
```

## Getting Started
```bash
python main.py --dataset-name DATASET_NAME(ex. cifar10) --device DEVICE(ex. cuda, cpu) #train
python main.py --dataset-name DATASET_NAME(ex. cifar10) --device DEVICE(ex. cuda, cpu) --load-from MODEL_PATH #test
```
