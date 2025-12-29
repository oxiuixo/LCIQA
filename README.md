# LCIQA

Official implementation of the paper:

**LCIQA: A Lightweight Contrastive-Learning Framework for Image Quality Assessment via Cross-Scale Consistency Minimization**

---

## Overview

we propose a Lightweight Contrastive-learning-based IQA (LCIQA) framework, designed to be efficiently trained on a single GPU without relying on ground truth data. 
This framework maintains a fixed vision backbone and focuses on optimizing the parameters of subsequent IQA heads through contrastive learning. 
To accommodate a lightweight framework, we incorporate a quality task adapter to eliminate semantic biases introduced by the features extracted from the fixed-parameter backbone. 
A coarse-to-fine contrastive learning strategy is then employed to train the quality regression module. 
Extensive experiments demonstrate the superior performance of our model in terms of both accuracy and complexity. 

---

## Experimental Settings

### ✅ Opinion Aware Setting (Available)

The current version of this repository **only includes the code for the Opinion Aware experimental setting**, where **opinion-related information is available during training**.

This implementation corresponds to the experiments reported under the *Opinion Aware* setting in the paper.

### ⏳ Opinion Unaware Setting (Coming Soon)

The code for the **Opinion Unaware experimental setting**, where **opinion information is not accessible to the model**, will be released in a future update.

---

@article{LCIQA2025,
  author={Feng, Chenxi and Min, Xiongkuo and Ye, Long and Yang, Yinghao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={LCIQA: A Lightweight Contrastive-Learning Framework for Image Quality Assessment via Cross-Scale Consistency Minimization}, 
  year={2025},
  volume={35},
  number={10},
  pages={9986-9999},
  doi={10.1109/TCSVT.2025.3558797}}
