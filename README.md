# Overview
Relevant code and files for Nina van Hoorn's 2025 undergraduate thesis. This thesis explores various dimensionality reduction techniques (mainly PCA, kernel PCA, and trained autoencoders) applied to molecular dynamics simulation data of the VCBC-A3F complex. 

To read the thsis, click here.

## How to navigate this code base
The thesis (and consequently, the code) is broken into four main categories. 

The code that is relevant to Chapter 4 "____" can be found in the directory _______. This includes the PTRAJ scripts to run PCA on the mdcrd files, the Jupyter notebook to visualize the PCA, the notebook used to perform K-Means clustering on the PCA, and the notebook used to calcualte the p-values.

The code that is relevant to Chapter 5 "_____" can be found in directories ______ and ______. These directories include the PTRAJ scripts to run PCA on exclusively the wild type mdcrds or the mutated mdcrds. Additionally, notebooks used to visualize these results can be found in each of the directories. 

The _______ directory contains the code relevant to Chapter 7 "________". This just contains three notebooks to run PCA on 1) the combined data, 2) PCs found from just the wild type data, and 3) PCs found from just the mutated simulation data. 

The code in directory _________ is for Chapter 8 "__________". This includes multipe subdirectories for exploring KPCA using different gamma values. 


## Awknowledgements
Thank you to the following who helped with the development of this code: Skidmore Professor Tom O'Connell, Skidmore Professor Aurelia Ball, and alumni of the Skidmore Computational Biophysics Lab.
