import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

import scipy.spatial.distance as S
from scipy.ndimage import gaussian_filter

import numpy as np
import heapq

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential, load_model

from sklearn.preprocessing import StandardScaler

import random

import os
import sys

epochs = int(sys.argv[1])
lr = float(sys.argv[2])

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(os.getenv('TF_GPU_ALLOCATOR'))


gpus = (tf.config.list_physical_devices('GPU'))


print(gpus)

# tf.debugging.set_log_device_placement(True)

traj_new = []
traj_new.append(md.load_netcdf("/data/nvanhoorn/data/GLU_1-8_first300.mdcrd",top="/data/nvanhoorn/data/GLU_nowat.prmtop", stride=10))

traj_new.append(md.load_netcdf("/data/nvanhoorn/data/WT_1-8_first300.mdcrd", top="/data/nvanhoorn/data/WT_nowat.prmtop", stride=10))


#traj_new[1].superpose(traj_new[1], atom_indices=traj_new[0].topology.select('backbone')) #align the proteins along the backbone
traj_new[0].atom_slice(traj_new[0].topology.select('backbone'), inplace=True)
traj_new[1].atom_slice(traj_new[1].topology.select('backbone'), inplace=True)
traj_new[1].superpose(traj_new[0])
traj_new[0].superpose(traj_new[0])
print('len traj_new = ', len(traj_new))

print('n frames = ', traj_new[0].n_frames) 


coordinates = traj_new[0].xyz
coordinatesWT = traj_new[1].xyz


print('length coordinates = ',  len(coordinates))


#coordinates = np.array([ np.concatenate((d, [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]])) for d in coordinates])
#coordinatesWT = np.delete(coordinatesWT, 907, 1)
#coordinatesWT = np.delete(coordinatesWT, 906, 1)
#coordinatesWT = np.delete(coordinatesWT, 905, 1)
#coordinatesWT = np.delete(coordinatesWT, 904, 1)
#coordinatesWT = np.delete(coordinatesWT, 903, 1)
#coordinatesWT = np.delete(coordinatesWT, 902, 1)
#coordinatesWT = np.delete(coordinatesWT, 901, 1)
#coordinatesWT = np.delete(coordinatesWT, 900, 1)
#coordinatesWT = np.delete(coordinatesWT, 899, 1)

#coordinates = np.delete(coordinates, 900, 1)
#coordinates = np.delete(coordinates, 899, 1)


# Calculate the split index
# split_index = len(coordinates) * 7 // 8  # 7/8ths of the array
#split_index = len(coordinates) * 3 // 8  # 7/8ths of the array
#end_index = len(coordinates) * 4 // 8  # 7/8ths of the array
split_index = len(coordinatesWT) * 7//8

# Slice the array
train_dataGLU = coordinates[:split_index]  # First 7/8ths of GLU
test_dataGLU = coordinates[split_index:]  # Last 1/8th of GLU

train_dataWT = coordinatesWT[:split_index]
test_dataWT = coordinatesWT[split_index:]  



# Print shapes to verify
print(train_dataGLU.shape) 
print(test_dataGLU.shape)  
print(train_dataWT.shape) 
print(test_dataWT.shape)  

train_data = np.concatenate((train_dataGLU, train_dataWT)) #combine GLU and WT data
test_data = np.concatenate((test_dataGLU, test_dataWT))

# train_data = train_dataGLU
# test_data = test_dataGLU

train_dataWT.shape

train_dataGLU.shape

train_data

del train_dataWT
del train_dataGLU
del test_dataWT
      
del test_dataGLU
del coordinates 
del coordinatesWT

train_data = train_data.reshape(train_data.shape[0], -1)  # Flatten to (n_frames, n_atoms * 3)


test_data = test_data.reshape(test_data.shape[0], -1)  # Flatten to (n_frames, n_atoms * 3)


scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)


encoding_dim = 3

# input_dims = 3 * (train_data[0].n_residues) #find the number of input dimensions by multiplying 3 by the number of residues
input_dims = train_data.shape[1]
print('input dims = ', input_dims)
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    input_prot = Input(shape=(input_dims,))

    drop = 0.5
#    encoded = Dense(1000, activation='relu')(input_prot)
#    encoded = Dropout(drop)(encoded)
    encoded = Dense(300, activation='relu')(input_prot)
    encoded = Dropout(drop)(encoded)
#    encoded = Dense(100, activation='relu')(encoded)
#    encoded = Dropout(drop)(encoded)
    encoded = Dense(30, activation='relu')(encoded)
    encoded = Dropout(drop)(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    decoded = Dense(30, activation='relu')(encoded)
    decoded = Dropout(drop)(decoded)
#    decoded = Dense(100, activation='relu')(decoded)
#    decoded = Dropout(drop)(decoded)
    decoded = Dense(300, activation='relu')(decoded)
    decoded = Dropout(drop)(decoded)
#    decoded = Dense(1000, activation='relu')(decoded)
#    decoded = Dropout(drop)(decoded)
    decoded = Dense(input_dims, activation='linear')(decoded)


    # this model maps an input to its reconstruction
    autoencoder = Model(inputs=input_prot, outputs=decoded)


    # this model maps an input to its encoded representation
    encoder = Model(inputs=input_prot, outputs=encoded)

    decoder = Model(inputs=encoded, outputs=decoded)


    optimizer = keras.optimizers.Adam(learning_rate=lr)

    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[keras.metrics.RootMeanSquaredError()])



print('************ autoencoder summary **************')
autoencoder.summary()


print('train_data ', train_data)
print('test data', test_data)

#epochs = 300
batch_size = 32

# https://discuss.ai.google.dev/t/getting-memory-error-when-training-a-larger-dataset-on-the-gpu/30575/14
from tensorflow.data import Dataset

#train_data = tf.convert_to_tensor(train_data)
#print('after conversion', train_data)

#train_data = Dataset.from_tensor_slices(train_data)

#with tf.device("CPU"):
#    train_data = Dataset.from_tensor_slices(train_data).shuffle(4*batch_size).batch(batch_size)
    # validate = Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    
print('************ calling fit ***************')

history = autoencoder.fit(train_data, train_data,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(test_data, test_data))


print('********** fit completed ***************')

train_loss = history.history['loss']
val_loss = history.history['val_loss']  

plt.figure(figsize=(12, 8))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Autoencoder Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("train-val-loss-"+str(epochs)+'-'+str(lr)+".png") 
# plt.show()


# ###### plt.figure(figsize=(12, 8))
# plt.plot(train_loss, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Autoencoder Training Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# In[ ]:


plt.figure(figsize=(12, 8))
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Autoencoder Validation Loss')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('val-loss-'+str(epochs)+'-'+str(lr)+'.png') 

# In[ ]:


plt.figure(figsize=(12, 8))
plt.plot(train_loss, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Autoencoder Loss')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('train-loss-'+str(epochs)+'-'+str(lr)+'.png') 


import datetime

now = datetime.datetime.now()
autoencoder.save('trained_autoencoder_epochs-'+str(epochs)+'-batchsize-'+str(batch_size)+'splitindex'+str(split_index)+'date-'+str(now)+'.saved')
encoder.save('encoder.keras')
decoder.save('decoder.keras')
autoencoder.save('autoencoder.keras')

# new = load_model('trained_autoencoders/100_epochs.keras')

encoded_test_prots = encoder.predict(test_data)

decoded_test_prots = decoder.predict(encoded_test_prots)

audoencoder_test = autoencoder.predict(test_data)

dims = len(encoded_test_prots[0])

x = []
y = []
z = []
for i in encoded_test_prots:
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])

print(dims)

plt.figure()
plt.scatter(x,y)
plt.savefig('scatterXY-'+str(epochs)+'-'+str(lr)+'.png') 
# plt.show()

plt.figure()
plt.scatter(x,z)
plt.savefig('scatterXZ-'+str(epochs)+'-'+str(lr)+'.png') 
# plt.show()

plt.figure()
plt.scatter(y,z)
plt.savefig('scatterYZ-'+str(epochs)+'-'+str(lr)+'.png') 
# plt.show()
#plt.show()


from mpl_toolkits import mplot3d
plt.figure()
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(x, y, z)
plt.savefig('scatterXYZ-'+str(epochs)+'-'+str(lr)+'.png') 
# plt.show()

# Calculate the reconstruction error on the test set
reconstructed_data = autoencoder.predict(test_data)
reconstruction_error = np.mean(np.square(reconstructed_data - test_data), axis=1)
print("Mean reconstruction error on the test set:", np.mean(reconstruction_error))


print(test_data - audoencoder_test)

message = "Network finished training. Mean reconstruction error on the test set:"+str(np.mean(reconstruction_error))
os.system("echo FinishedTraining | mail -s NetworkUpdate nvanhoorn@skidmore.edu")
