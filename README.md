# Overview
Relevant code and files for Nina van Hoorn's 2025 undergraduate thesis. This thesis explores various dimensionality reduction techniques (mainly PCA, kernel PCA, and trained autoencoders) applied to molecular dynamics simulation data of the VCBC-A3F complex. 


## How to navigate this code base
The thesis (and consequently, the code) is broken into four main categories. 

The code that is relevant to **"Chapter 4: Identifying Conformational Clusters Using PTRAJ PCA"** can be found in the directory PCA_analysis/PTRAJ_PCA_combined_data/. This includes the PTRAJ scripts to run PCA on the mdcrd files, the Jupyter notebook to visualize the PCA, the notebook used to perform K-Means clustering on the PCA, and the notebook used to calculate the p-values.

The code that is relevant to **"Chapter 5: Further Exploration of PTRAJ PCA"** can be found in directories PCA_analysis/PTRAJ_PCA_just_WT_data/ and PCA_analysis/PTRAJ_just_K50E_data/. These directories include the PTRAJ scripts to run PCA on exclusively the wild type mdcrds or the mutated mdcrds. Additionally, notebooks used to visualize these results can be found in each of the directories. 

The PCA_analysis/Scikit-Learn_PCA/ directory contains the code relevant to **"Chapter 6: Visualizing Complex Motions Using Scikit-Learn PCA"**. This just contains three notebooks to run PCA on 1) the combined data, 2) PCs found from just the wild type data, and 3) PCs found from just the mutated simulation data. 

The code in directory KPCA_analysis/ is for **"Chapter 8: Reducing Dimensionality Through Kernel PCA"**. This includes multiple subdirectories for exploring KPCA using different gamma values. Additionally, the original KPCA which included scaling the values is in the kpca_scripts_with_scaling/ directory, though this analysis was not used in the thesis and is supplemental. Each of these subdirectories include five directories for each of the kernel functions. Most of these failed to run to completion, which is why the RBF kernel and the cosine kernel are the only two explored in the thesis. 

The directory autoencoder_files/ contains the autoencoder training code and visualization code for **"Chapter 8: Visualizing the Latent Space Through an Autoencoder."** Subdirectories are for the different autoencoders trained and include code to train the networks, make figures, as well as the figures made. There is an additional directory here which explores a shallow ReLU network that was trained for 500 epochs; this was not included in the thesis because of time constraints. 

Lastly, the code to make the contact maps from **"Chapter 9: The Effects of the Mutation"** is in the directory contact_maps/.

There is some additional code that that is available under miscellaneous. This includes code to generate the general PCA figures used in the thesis to visualize how PCA works conceptually (Figures 5 and 6). There is also code which manually computes the cosine similarity of the wild type and mutated simulation data from a few different times throughout simulations. This was created to ensure the proteins were aligned properly (this was in response to an earlier, flawed Cosine KPCA result), but was not addressed in the thesis. 


## Acknowledgements
Thank you to the following who helped with the development of this code: Skidmore Professor Tom O'Connell, Skidmore Professor Aurelia Ball, and alumni of the Skidmore Computational Biophysics Lab.

ChatGPT was also used at times, mainly for assisting with debugging, creating more effective Matplotlib figures, and using libraries (ie. mdtraj and Pandas).
