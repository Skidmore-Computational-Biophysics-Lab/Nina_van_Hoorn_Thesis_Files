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

pdb = md.load_pdb("/data/nvanhoorn/data/WT_tleap.pdb")

#traj_new[1].superpose(traj_new[1], atom_indices=traj_new[0].topology.select('backbone')) #align the proteins along the backbone
traj_new[0].atom_slice(traj_new[0].topology.select('backbone'), inplace=True)
traj_new[1].atom_slice(traj_new[1].topology.select('backbone'), inplace=True)
pdb.atom_slice(traj_new[1].topology.select('backbone'), inplace=True)
traj_new[1].superpose(traj_new[0])
traj_new[0].superpose(traj_new[0])
pdb.superpose(traj_new[0])
print('len traj_new = ', len(traj_new))

print('n frames = ', traj_new[0].n_frames) 


coordinates = traj_new[0].xyz
coordinatesWT = traj_new[1].xyz
coordinatesST = pdb.xyz


print('length coordinates = ',  len(coordinates))

# Calculate the split index
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


train_dataWT.shape

train_dataGLU.shape

train_data

del train_dataWT
del train_dataGLU
del test_dataWT
      
del test_dataGLU

train_data = train_data.reshape(train_data.shape[0], -1)  # Flatten to (n_frames, n_atoms * 3)
test_data = test_data.reshape(test_data.shape[0], -1)  # Flatten to (n_frames, n_atoms * 3)
GLU_data = coordinates.reshape(coordinates.shape[0], -1)
WT_data = coordinatesWT.reshape(coordinatesWT.shape[0], -1)
ST_data = coordinatesST.reshape(coordinatesST.shape[0], -1)

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)
GLU_data = scaler.transform(GLU_data)
WT_data = scaler.transform(WT_data)
ST_data = scaler.transform(ST_data)

GLU_data = GLU_data[::25]
WT_data = WT_data[::25]

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
    
#print('************ calling fit ***************')

#history = autoencoder.fit(train_data, train_data,
#    epochs=epochs,
#    batch_size=batch_size,
#    shuffle=True,
#    validation_data=(test_data, test_data))


print('********** fit completed ***************')

#train_loss = history.history['loss']
#val_loss = history.history['val_loss']  
#
#plt.figure(figsize=(12, 8))
#plt.plot(train_loss, label='Training Loss')
#plt.plot(val_loss, label='Validation Loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.title('Autoencoder Training and Validation Loss')
#plt.legend()
#plt.grid(True)
#plt.savefig("train-val-loss-"+str(epochs)+'-'+str(lr)+".png") 
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


#plt.figure(figsize=(12, 8))
#plt.plot(val_loss, label='Validation Loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.title('Autoencoder Validation Loss')
#plt.legend()
#plt.grid(True)
# plt.show()
#plt.savefig('val-loss-'+str(epochs)+'-'+str(lr)+'.png') 
#
# In[ ]:


#plt.figure(figsize=(12, 8))
#plt.plot(train_loss, label='Loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.title('Autoencoder Loss')
#plt.legend()
#plt.grid(True)
# plt.show()
#plt.savefig('train-loss-'+str(epochs)+'-'+str(lr)+'.png') 


import datetime

#now = datetime.datetime.now()
#autoencoder.save('trained_autoencoder_epochs-'+str(epochs)+'-batchsize-'+str(batch_size)+'splitindex'+str(split_index)+'date-'+str(now)+'.saved')
#encoder.save('encoder.keras')
#decoder.save('decoder.keras')
#autoencoder.save('autoencoder.keras')

encoder = load_model('../encoder.keras')

#encoded_test_prots = encoder.predict(test_data)
#decoded_test_prots = decoder.predict(encoded_test_prots)
#audoencoder_test = autoencoder.predict(test_data)

print('***********Load Succesful***********')

encodedGLU = encoder.predict(GLU_data)
encodedWT = encoder.predict(WT_data)
encodedST = encoder.predict(ST_data)

print('encodedGLU:',encodedGLU[:20:])
print('encodedWT:',encodedWT[:20:])
print('encodedST:',encodedST)

print('*********** Predictions made ***********')

xG, yG, zG = zip(*encodedGLU)
xW, yW, zW = zip(*encodedWT)
xS, yS, zS = zip(*encodedST)

#xG = []
#yG = []
#zG = []

#xW = []
#yW = []
#zW = []

#xS = []
#yS = []
#zS = []

#for i in encodedGLU:
#    xG.append(i[0])
#    yG.append(i[1])
#    zG.append(i[2])
#for j in encodedWT:
#    xW.append(j[0])
#    yW.append(j[1])
#    zW.append(j[2])
#for l in encodedST:
#    xS.append(l[0])
#    yS.append(l[1])
#    zS.append(l[2])

plt.figure()
plt.scatter(xG,yG, color='r',label='K50E mut',alpha=0.5)
plt.scatter(xW,yW, color='b', label='WT',alpha=0.5)
plt.scatter(xS,yS, color='black', marker='X', label="Starting Struct")
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('scatterXY-Lessdata.png') 
# plt.show()

plt.figure()
plt.scatter(xG,zG, color='r',label='K50E mut',alpha=0.5)
plt.scatter(xW,zW, color='b',label='WT',alpha=0.5)
plt.scatter(xS,zS, color='black', marker='X', label="Starting Struct")
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 3')
plt.savefig('scatterXZ-Lessdata.png') 
# plt.show()

plt.figure()
plt.scatter(yG,zG, color='r',label='K50E mut',alpha=0.5)
plt.scatter(yW,zW, color='b',label='WT',alpha=0.5)
plt.scatter(yS,zS, color='black', marker='X', label="Starting Struct")
plt.xlabel('Dimension 2')
plt.ylabel('Dimesnion 3')
plt.savefig('scatterYZ-Lessdata.png') 


xmin = min(min(xG), min(xW), min(xS))
xmax = max(max(xG), max(xW), max(xS))
ymin = min(min(yG), min(yW), min(yS))
ymax = max(max(yG), max(yW), max(yS))
zmin = min(min(zG), min(zW), min(zS))
zmax = max(max(zG), max(zW), max(zS))

from mpl_toolkits import mplot3d
plt.figure()
ax = plt.axes(projection ="3d")
plt.scatter(xG,yG,zG, color='r',label='K50E mut',alpha=0.4)
plt.scatter(xW,yW,zW, color='b',label='WT',alpha=0.4)
plt.scatter(xS,yS,zS, color='black', marker='X', label="Starting Struct", alpha=0.5)
ax.set_xlim([xmin,xmax])
ax.set_ylim([ymin,ymax])
ax.set_zlim([zmin,zmax])
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.savefig('scatterXYZ-Lessdata.png') 
# plt.show()


plt.figure()
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(xG,yG,zG,color='r',label='K50E', alpha=0.4, s=20)
ax.scatter(xW,yW,zW,color='b',label='WT',alpha=0.4,s=20)
ax.scatter(xS,yS,zS,color='black',label='SS',marker='X')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('final3D.png')


plt.figure()
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(xG,yG,zG,color='r',label='K50E', alpha=0.4, s=20)
ax.scatter(xW,yW,zW,color='b',label='WT', alpha=0.4, s=20)
ax.scatter(xS,yS,zS,color='black',label='SS', marker='X')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=90,azim=90)
plt.savefig('3d_90-90.png')

plt.figure()
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(xG,yG,zG,color='r',label='K50E', alpha=0.4, s=20)
ax.scatter(xW,yW,zW,color='b',label='WT', alpha=0.4, s=20)
ax.scatter(xS,yS,zS,color='black',label='SS', marker='X')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=90,azim=50)
plt.savefig('3d_90-50.png')

plt.figure()
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(xG,yG,zG,color='r',label='K50E', alpha=0.4, s=20)
ax.scatter(xW,yW,zW,color='b',label='WT', alpha=0.4, s=20)
ax.scatter(xS,yS,zS,color='black',label='SS', marker='X')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=50,azim=90)
plt.savefig('3d_50-90.png')

'''
import plotly.graph_objects as go

# Create traces for the data
trace_GLU = go.Scatter3d(
    x=xG, y=yG, z=zG,
    mode='markers',
    marker=dict(
        color='red',
        size=5,
        opacity=0.6
    ),
    name='K50E mut'
)

trace_WT = go.Scatter3d(
    x=xW, y=yW, z=zW,
    mode='markers',
    marker=dict(
        color='blue',
        size=5,
        opacity=0.6
    ),
    name='WT'
)

trace_ST = go.Scatter3d(
    x=xS, y=yS, z=zS,
    mode='markers',
    marker=dict(
        color='black',
        size=5,
        symbol='x',
        opacity=0.6
    ),
    name="Starting Struct"
)

# Define layout for the plot
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='Dimension 1'),
        yaxis=dict(title='Dimension 2'),
        zaxis=dict(title='Dimension 3')
    ),
    title="3D Scatter Plot"
)

# Create the figure
fig = go.Figure(data=[trace_GLU, trace_WT, trace_ST], layout=layout)

fig.write_image("3d_scatter_plot.png")  # Save the plot as a PNG file
'''

os.system("echo Plots Made | mail -s NetworkUpdate nvanhoorn@skidmore.edu")
