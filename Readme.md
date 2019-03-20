# BWGAN Pytorch

Unofficial Pytorch implementation of Banach WGAN https://arxiv.org/pdf/1806.06621.pdf

[Our report](https://clck.ru/FPJWN)

![Example, s = -2, p = 2](images/-2_2.png?raw=True "s = -2, p = 2")

Training (following arguments are required, see ```python train.py -h``` for further information):
```
python train.py --dataset DATASET --name EXPERIMENT --cuda ID --s S --p P
```
FID and Inception Score calculating:
```
python fid.py --original PATH_TO_ORIGINAL_DATASET --generated PATH_TO_GENERATED --models PATH_TO_TRAINED_MODELS --cuda ID

python inception.py --path PATH_TO_DATASET --cuda ID
```

Requirements:
 * Pytorch==0.4.1
 * Latest CUDA GPU for acceleration
