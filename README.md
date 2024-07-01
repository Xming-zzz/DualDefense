# Dual Defense
This is the code for paper: 
"Dual Defense: Adversarial, Traceable, and Invisible Robust Watermarking Against Face Swapping," in IEEE Transactions on Information Forensics and Security, vol. 19, pp. 4628-4641, 2024. 
https://ieeexplore.ieee.org/document/10486948

# INTRODUCTION


Our work proposes a novel active defense method Dual Defense based on robust adversarial watermark. Dual Defense embeds a robust adversarial watermark into the carrier facial image at one time, thereby destroying the deep face swapping model while tracking the copyright of the facial image.

# Model Architecture

## Our Model 

![image](https://github.com/Xming-zzz/DualDefense/assets/55829247/ba9afe39-c9b2-40ab-afd1-ddbfa051042f)

## Victim Model

You can download the faceswap model through the official link: https://github.com/Oldpan/Faceswap-Deepfake-Pytorch

## Face Recognition Model

The FaceNet model is used for performance evaluation and can be downloaded through the official link: https://github.com/davidsandberg/facenet

# Requirements
The core packages we use in the project and their version information are as follows:

kornia==0.6.12

numpy==1.22.3

pandas==1.3.5

torch==1.12.0

torchvision==0.13.0

## Train

```
python train.py
```

## Test

```
python test.py
```
