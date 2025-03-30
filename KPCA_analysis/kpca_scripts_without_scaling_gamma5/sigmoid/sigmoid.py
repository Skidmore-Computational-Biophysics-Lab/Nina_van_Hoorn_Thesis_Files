import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler


traj_new = []
traj_new.append(md.load_netcdf("/data/nvanhoorn/A3F_newstructure/VCBC_A3F_GLU_mut/redoing_everything/mdcrd/VCBC-A3F_GLU_mut_sims1-8_first300_noBox.mdcrd", top="/data/nvanhoorn/A3F_newstructure/VCBC_A3F_GLU_mut/redoing_everything/VCBC-A3F_GLU_mut_nowat.prmtop", stride=10))

traj_new.append(md.load_netcdf("/data/nvanhoorn/A3F_newstructure/VCBC_A3F_GLU_mut/redoing_everything/WT_files/VCBC-A3F_WT_sims1-8_first300_noBox.mdcrd", top="/data/nvanhoorn/A3F_newstructure/VCBC_A3F_GLU_mut/redoing_everything/WT_files/VCBC-A3F_WT_nowat.prmtop", stride=10))


starting_struct = md.load_pdb("/data/nvanhoorn/A3F_newstructure/VCBC_A3F_GLU_mut/redoing_everything/WT_files/VCBC-A3F_WT_tleap.pdb")

traj_new[0].atom_slice(traj_new[0].topology.select('backbone'), inplace=True)
traj_new[1].atom_slice(traj_new[1].topology.select('backbone'), inplace=True)
starting_struct.atom_slice(traj_new[1].topology.select('backbone'), inplace=True)

traj_new[1].superpose(reference=traj_new[0])
traj_new[0].superpose(reference=traj_new[0])
starting_struct.superpose(reference=traj_new[0])

coordinates = traj_new[0].xyz
coordinatesWT = traj_new[1].xyz
Startcoordinates = starting_struct.xyz


split_index = len(coordinates) * 7 // 8  # 7/8ths of the array

train_data = np.concatenate((coordinates, coordinatesWT))

# Print shapes to verify
print('training data shape', train_data.shape) 

train_data = train_data[::25] #cut down the training data for reduce runtime

print('reduced training data shape', train_data.shape) 

train_data = train_data.reshape(train_data.shape[0], -1)  # Flatten to (n_frames, n_atoms * 3) #FROM AUTOENCODER
start_struct_data = Startcoordinates.reshape(Startcoordinates.shape[0], -1)


#scaler = StandardScaler()
#train_data = scaler.fit_transform(train_data)

print("******************** Starting KPCA ********************")


kernel_pca_sigmoid = KernelPCA(
    n_components=None, kernel="sigmoid", fit_inverse_transform=True, alpha=0.1, gamma=5
)
# sigmoid_kernel_pca = kernel_pca_sigmoid.fit(train_data).transform(test_data)

sigmoid_kernel_pca = kernel_pca_sigmoid.fit(train_data)
sigmoid_train_kernel_pca = sigmoid_kernel_pca.transform(train_data)

print("******************** KPCA Finished ********************")

fig, (orig_data_ax, kernel_pca_proj_ax) = plt.subplots(
    ncols=2, figsize=(14, 4)
)
orig_data_ax.scatter(train_data[:, 0], train_data[:, 1])
orig_data_ax.set_ylabel("Feature #1")
orig_data_ax.set_xlabel("Feature #0")
orig_data_ax.set_title("Training data")
kernel_pca_proj_ax.scatter(sigmoid_train_kernel_pca[:, 0], sigmoid_train_kernel_pca[:, 1])
kernel_pca_proj_ax.set_ylabel("Principal component #1")
kernel_pca_proj_ax.set_xlabel("Principal component #0")
_ = kernel_pca_proj_ax.set_title("Projection of training data\n using KernelPCA")

fig.savefig('TrainingData.png', bbox_inches = "tight", dpi= 2000)




GLU = coordinates.reshape(coordinates.shape[0], -1) 
WT = coordinatesWT.reshape(coordinatesWT.shape[0], -1)
#GLU = scaler.transform(GLU)
#WT = scaler.transform(WT)
ST = (start_struct_data)

GLU = GLU[::25] #cut down the data projected 
WT = WT[::25]  #cut down the data projected 

sigmoid_GLU_kernel_pca_normed = sigmoid_kernel_pca.transform(GLU)
sigmoid_WT_kernel_pca_normed = sigmoid_kernel_pca.transform(WT)
sigmoid_ST_kernel_pca_normed = sigmoid_kernel_pca.transform(ST)

plt.figure(figsize=(12, 8))
plt.scatter(sigmoid_GLU_kernel_pca_normed[:, 0], sigmoid_GLU_kernel_pca_normed[:, 1])
plt.ylabel("Feature #1")
plt.xlabel("Feature #0")
plt.title("GLU data")
plt.savefig('GLUData.png', bbox_inches = "tight", dpi= 2000)


plt.figure(figsize=(12, 8))
plt.scatter(sigmoid_WT_kernel_pca_normed[:, 0], sigmoid_WT_kernel_pca_normed[:, 1])
plt.ylabel("Feature #1")
plt.xlabel("Feature #0")
plt.title("WT data")
plt.xlim([-1,1])

plt.figure(figsize=(12, 8))
plt.scatter(sigmoid_GLU_kernel_pca_normed[:, 0], sigmoid_GLU_kernel_pca_normed[:, 1], color='r', alpha=0.5, label='K50E MUT')
plt.scatter(sigmoid_WT_kernel_pca_normed[:, 0], sigmoid_WT_kernel_pca_normed[:, 1], color='b', alpha=0.5, label='WT')
plt.scatter(sigmoid_ST_kernel_pca_normed[:, 0], sigmoid_ST_kernel_pca_normed[:, 1], marker='X',color='black', label='Starting Struct')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.title('PCA Transformed Data')
plt.savefig('WTandGLUData.png', bbox_inches = "tight", dpi= 2000)

print("******************** Finished ********************")

                                                          
