# DiPriMeFlip Forest
Differentially Private Median Forests (with several DP mechanisms)

## File descriptions:
### Classes:
  * **DiPrime.py:** Class definition for DiPriMe (set _max_features_ to 1) and DiPriMeExp for regression
  * **DiPrime_Class.py:** Class definition for DiPriMe and DiPriMeExp for classification
  * **DiPrimeFlip.py:** Class definition of DiPriMeFlip for regression - uses the [permute-and-flip mechanism](https://arxiv.org/abs/2010.12603) [1]
  * **DiPrimeFlip_Class.py:** Class definition of DiPriMeFlip for classification
  * **RandForest_DP.py:** Class definition of differentially private forests with extremely random splits for regression
  * **RandForst_DP_Class.py:** Class definition of differentially private forests with extremely random splits for classification

### Baselines:
  * **DP-DF:** Contains files for the implementation of [DP-DF](https://www.researchgate.net/profile/Md_Islam61/publication/308802652_A_Differentially_Private_Decision_Forest/links/57f3718508ae8da3ce51b330/A-Differentially-Private-Decision-Forest.pdf) [2]
  * **dprf.py:** Implementation of [DP-RF](https://ieeexplore.ieee.org/abstract/document/6968348?casa_token=PoyO5iQL10cAAAAA:lz7HfukwBVWyUjTUll_Eg9MYjEcXKZTaIdGqpeqb3E-gOX6qF5khqeTqADWb_l8NopxaW3LmnEO6) [3]
  * **dprf_Reg.py:** Regression analogue of DP-RF

### Datasets:
Contains datasets used for our experiments

### DiPriMeFlip_Results:
Results from experiments in Jupyter notebooks

### Notebooks:
  * **MedianTree_UCI.ipynb:** Experiments to compare performance of DiPriMe against various tree-based methods for regression and classification datasets
  * **MedianTree_param.ipynb:** Experiments to observe variation in performance of algorithms with change in hyperparameter values

## Overview of methodology:
Existing methodologies query optimal splits privately or pick random splits at each node. This leads to many low-occupancy or empty leaf nodes in the resulting tree. Instead, we use private versions of medians as splits at each node to achieve a more balanced tree structure, thereby reducing the possibility of noise overpowering the signal at each leaf node. We explore multiple DP mechanisms and compare their utility to existing tree ensemble algorithms.
<br /><br />

**References**:

[1] McKenna, R., & Sheldon, D. (2020). Permute-and-Flip: A new mechanism for differentially private selection. arXiv preprint arXiv:2010.12603.

[2] Fletcher, S., & Islam, M. Z. (2015). A Differentially Private Decision Forest. AusDM, 15, 99-108.

[3] Patil, A., & Singh, S. (2014, September). Differential private random forest. In 2014 International Conference on Advances in Computing, Communications and Informatics (ICACCI) (pp. 2623-2630). IEEE.