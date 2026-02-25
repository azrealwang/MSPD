# Minimal Cascade Gradient Smoothing for Fast Transferable Preemptive Adversarial Defense

[PDF](https://arxiv.org/pdf/2407.15524)

Adversarial attacks persist as a major challenge in deep learning. While training- and test-time defenses are well-studied, they often reduce clean accuracy, incur high cost, or fail under adaptive threats. In contrast, preemptive defenses, which perturb media before release, offer a practical alternative but remain slow, model-coupled, and brittle.
* We propose the **Minimal Sufficient Preemptive Defense (MSPD)**, a fast, transferable framework that defends against future attacks without access to the target model or gradients.
* MSPD is driven by **Minimal Cascade Gradient Smoothing (MCGS)**, a two-epoch optimization paradigm executed on a surrogate backbone. This defines a minimal yet effective regime for robust generalization across unseen models and attacks.
* To evaluate adaptive robustness, we introduce **Preemptive Reversion**, the first white-box diagnostic attack that cancels preemptive perturbations under full gradient access.

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

## User Study

We provide the full user study questionnaire in `UserStudy.png`.

## Citation
```
@article{wang2026minimal,
  title={Minimal Cascade Gradient Smoothing for Fast Transferable Preemptive Adversarial Defense},
  author={Wang, Hanrui and Chang, Ching-Chun and Lu, Chun-Shien and Kao, Ching-Chia, Shuo Wang, and Echizen, Isao},
  journal={arXiv preprint arXiv:2407.15524},
  year={2026}
}
```

## Contact
If you have any questions about our work, please do not hesitate to contact us by email.

Hanrui Wang: hanrui_wang@nii.ac.jp
