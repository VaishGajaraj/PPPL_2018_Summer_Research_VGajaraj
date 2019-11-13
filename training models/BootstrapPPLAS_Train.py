import pandas as pd
from nubeamnet import NubeamNet
from get_data import dataset
import numpy as np
import get_data as gd
import pickle
import MDSplus as mds
import nubeamnet as nb
import os
import matplotlib.pyplot as plt

dataset_path = './datasets/magdif_PPLAS/dataset.p'
 
print("Loading dataset...")
ds = pickle.load(open(dataset_path))
nkeep = 20
X_profiles = ['TE', 'NE', 'Q', 'DIFB', 'PPLAS']
X_n_keep = [5,5,5,5,5,7]
y_profiles = ['CURBSEPS', 'CURBS' , 'CURBSSAU', 'CURBSWNC', 'CURGP', 'ETA_SNC', 'ETA_SP', 'ETA_SPS', 'ETA_WNC']
y_n_keep = [12,12,12,12,12,4,4,4,4]
nprofiles = len(y_profiles)+len(X_profiles)
X_scalars=['Shot','ID','ZEFFC', 'R0','ELONG','PCURC','AMIN','BZXR','TRIANGU','TRIANGL']
y_scalars=['Shot','ID']

apply_pinj_filtering=False
#load the data
#load 1 batch while testing the script
#batch_data = net.load_batch_data(split='training', batch_numbers=np.arange(ds.n_batches['training'])+1,apply_pinj_filtering=False)
#load all batches when running for real
#batch_data = net.load_batch_data(split='training', batch_numbers=np.arange(1)+1,apply_pinj_filtering=False)

	#find principle components of profiles

#net.batch_train_pca(batch_data, partial_fit=False)


net_save_path = "hidden_layer5/"


print("Loading data...")

dummy_net = nb.NubeamNet(ds, early_stopping=False, n_nn=1, learning_rate_init=0.001, hidden_layer_sizes=(15),
              ensemble_exclude_fraction=0.2, X_scalars = X_scalars, y_scalars=y_scalars, X_n_keep = X_n_keep, y_n_keep=y_n_keep, X_profiles = X_profiles, y_profiles=y_profiles)  #dummy net object used to load the data once

batch_data = dummy_net.load_batch_data(split='training', batch_numbers=np.arange(ds.n_batches['training'])+1, apply_pinj_filtering=False) #add apply_pinj_filtering=False
#y_profiles=['PFI','CURB','PBE','PBI','TQBE','TQBI','BDENS'] #update X_profiles, y_profiles, X_scalars, y_scalars

print("Beginning scan...")
nets = [nb.NubeamNet(ds, early_stopping=False, n_nn=8, learning_rate_init=0.001, hidden_layer_sizes=sizes,
              ensemble_exclude_fraction=0.2, X_scalars = X_scalars, y_scalars=y_scalars, X_n_keep = X_n_keep, y_n_keep=y_n_keep, X_profiles = X_profiles, y_profiles=y_profiles) for sizes in [(25,)]] #update parameters, list of sizes
for j, net in enumerate(nets):
	print("Training net "+str(j))
	net.batch_train_pca(batch_data, partial_fit=False)
	data_df = net.form_data_df(batch_data)
	print("Training normalization...")
	net.batch_train_normalization(data_df, partial_fit=False)
	print("Training neural nets...")
	net.train_ensemble(data_df, validation_data=None, partial_fit=False)
	net_save_name = 'nn_1layer_magdifPPLAS_scan'+str(j) #update prefix to name: 1layer, 2layer, 3layer
	net.save(net_save_name,net_save_path)
	print(len(net.training_shots))


#data_df = net.form_data_df(batch_data)

#net.batch_train_normalization(data_df,partial_fit=False)

#net.train_ensemble(data_df, validation_data=None,partial_fit=False)
#net.save(net_save_name,net_save_path)
print("Training complete.")

