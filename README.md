# RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models
[![arXiv](https://img.shields.io/badge/arXiv-2411.04097-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2411.04097)
[![License](https://img.shields.io/github/license/stanford-aimi/ravl?style=for-the-badge)](LICENSE)

This repository contains the official PyTorch implementation for [RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models](https://arxiv.org/abs/2411.04097) (NeurIPS 2024).

![Overview](assets/img.png "")

## 🧵 What is RaVL?
Fine-tuned vision-language models (VLMs) often capture spurious correlations between image features and textual attributes, resulting in degraded zero-shot performance at test time. We introduce **RaVL**, which can analyze a VLM and identify specific image features that the model has learned to spuriously correlate with a textual attribute. 

For additional details, please refer to our [paper](https://arxiv.org/abs/2411.04097), our [demo notebook](https://github.com/Stanford-AIMI/RaVL/blob/master/demo_mnist.ipynb), and the documentation below. 

## ⚡️ Installation
Use the following commands to clone and install this repository. Confirm that PyTorch and torchvision are installed on your system.
```python
git clone https://github.com/Stanford-AIMI/RaVL.git
cd ravl
pip install -e .
```
Then, create a file ```.env``` with the path to the package root (refer to ```.env_example``` for an example).

## ⚙️ Discovering Spurious Correlations with RaVL
Code for using RaVL to discover spurious correlations in VLMs is provided in ```ravl/discover.py```. 

For a detailed walkthrough on running RaVL on an example setting, please refer to our demo notebook ```demo_mnist.ipynb```.


## 🧪 Designing Controlled Evaluations 
In this work, we propose a large-scale evaluation framework where the ground-truth spurious correlations learned by VLMs are known and annotated in advance; this can then enable us to determine whether the features discovered by RaVL accurately align with the ground truth. 

Our evaluation framework operates by artificially inducing spurious correlations in the VLM fine-tuning data. Then, given the known pre-defined spurious correlation and a fine-tuned VLM that learned the desired spurious correlation, we can quantitatively evaluate the extent to which RaVL discovers the correlation. 

Here, we provide code for constructing evaluation settings using MNIST. 

### Step 1: Create Base MNIST Dataset
First, run the following command to generate a base MNIST dataset in our desired format. Each image in the base dataset consists of four quadrants, with one quadrant containing an MNIST digit. A separate quadrant contains a red rectangle with 50% probability. The remaining quadrants are empty.
```python
python3 -m build_eval.create_base_mnist
```

### Step 2: Create Evaluation Settings
Then, run the following command to generate a set of vision-language fine-tuning datasets with artifically induced spurious correlations between a visual feature (red rectangle, in this case) and a textual attribute (MNIST digit, in this case). These fine-tuning datasets are sampled from the base MNIST dataset such that the desired spurious correlation exists. The size of the fine-tuning dataset and the strength of the spurious correlation can be customized in the  ```get_params()``` function in ```build_eval/generate_spurious_mnist.py```. 
```python
python3 -m build_eval.generate_spurious_mnist
```
The command above also generates zero-shot classification datasets for evaluation. Evaluation datasets are designed to resemble real-world settings by ensuring that the spurious correlation does not exist. 

### Step 3: Finetune VLM
A VLM (e.g. CLIP) can be fine-tuned on the generated fine-tuning dataset and evaluated using the generated evaluation dataset. The evaluation setting is valid if the VLM learns the spurious correlation; in other words, performance gaps with respect the presence and absence of the rectangle must be observed, as discussed in Section 3.2 of our paper.

### Examples
Below, we provide some image-text examples from a fine-tuning dataset with an artificially induced spurious correlation between red rectangles and the MNIST digit "nine". As shown by these examples, it is clear that the red rectangle and the MNIST digit "nine" frequently co-occur, which means that a VLM fine-tuned using this data is likely to capture this spurious correlation. We provide code in ```build_eval/visualize_data.ipynb``` to visualize evaluation settings.

Examples with the textual attribute "nine": 
![Fig1](assets/data_img1.png "")

Examples without the textual attribute "nine": 
![Fig2](assets/data_img2.png "")


## 📎 Citation
If you find this repository useful for your work, please cite the following paper:

```
@inproceedings{varma2024ravl,
title={RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models},
author={Maya Varma and Jean-Benoit Delbrouck and Zhihong Chen and Akshay S Chaudhari and Curtis Langlotz},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=UFRZHFYW8e}
}
```

This repository was inspired by [ViLLA](https://github.com/StanfordMIMI/villa) and [ViLMedic](https://github.com/jbdel/vilmedic).
