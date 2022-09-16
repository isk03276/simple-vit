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
python main.py --dataset-name DATASET_NAME(ex. cifar10) --device DEVICE(ex. cuda, cpu) --load-from MODEL_PATH --load-model-config #test
```


## Results
**- CIFAR-10**  
<img src="https://user-images.githubusercontent.com/23740495/190575759-317fe1fc-57a0-4771-abb1-41925d72e051.png" width="70%" height="70%"/>

