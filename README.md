# Minimal Cascade Gradient Smoothing for Low-Latency Transferable Preemptive Adversarial Defense

This is a demo of our proposed preemptive defense, **MSPD**, and adaptive reversion attack, **Preemptive Reversion**, on the CIFAR-10 dataset.

****
## Contents
* [Main Requirements](#Main-Requirements)
* [Installation](#Installation)
* [Models](#Models)
* [Usage](#Usage)

****

## Main Requirements

  * **Python (3.9.32)**
  * **torch (2.1.2+cu118)**
  * **torchvision (0.16.2+cu118)**
  * **[RobustBench](https://github.com/RobustBench/robustbench)** - providing classifier, backbone, and target models
  * **[AutoAttack](https://github.com/fra31/auto-attack)** - for evaluation only
  
  
  The versions in `()` have been tested.

## Installation
Install PyTorch
With GPU:

```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```
With CPU:
```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
```
Install RobustBench
```
pip install git+https://github.com/RobustBench/robustbench.git
```
Install AutoAttack
```
pip install git+https://github.com/fra31/auto-attack
```

## Models
This is a model list used as backbones and targets in our paper. The checkpoints will be automtically downloaded.

| # | Architecture | Configuration | Comment |
|:---:|:---:|:---:|:---:|
| 0 | PreActResNet-18 | Gowal2021Improving_R18_ddpm_100m | Our **backbone** |
| 1 | WideResNet-94-16 | Bartoldson2024Adversarial_WRN-94-16 | SOTA robustness |
| 2 | RaWideResNet-70-16 | Peng2023Robust |  |
| 3 | WideResNet-70-16 | Wang2023Better_WRN-70-16 |  |
| 4 | WideResNet-34-10 | Rade2021Helper_extra |  |
| 5 | WideResNet-28-10 | Xu2023Exploring_WRN-28-10 |  |
| 6 | PreActResNet-18 | Wong2020Fast | Same architecture but<br>different weights with backbone |
| 7 | ResNeSt-152 | Sehwag2021Proxy_ResNest152 |  |
| 8 | ResNet-50 | Chen2020Adversarial |  |
| 9 | ResNet-18 | Addepalli2022Efficient_RN18 |  |
| 10 | XCiT-L12 | Debenedetti2022Light_XCiT-L12 | Transformer |

## Usage
AutoAttack on clean images
```
python attack_white.py --attack AutoAttack --version standard --norm Linf --eps 8 --data cifar10 --model Bartoldson2024Adversarial_WRN-94-16 --input imgs/cifar10/clean-0 --output imgs/aa8_clean_Bart
```
Run preemptive defense - MSPD
```
python MSPD.py --eps 8 --N 20 --data cifar10 --backbone Gowal2021Improving_R18_ddpm_100m --input imgs/cifar10/clean-0 --output imgs/mspd
```
AutoAttack on defended images
```
python attack_white.py --attack AutoAttack --version standard --norm Linf --eps 8 --data cifar10 --model Bartoldson2024Adversarial_WRN-94-16 --input imgs/mspd --output imgs/aa8_mspd_Bart
```
Run preemptive reversion
```
python MSPD.py --reversion --eps 8 --N 20 --data cifar10 --backbone Gowal2021Improving_R18_ddpm_100m --input imgs/mspd --output imgs/R_mspd
```