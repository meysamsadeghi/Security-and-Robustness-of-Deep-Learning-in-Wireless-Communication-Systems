# test
#import matplotlib.pyplot as plt   
import numpy as np
import tensorflow as tf


#import time
#import copy

k = 4
n = 7
seed = 0

from All_Autoencoder_Classes_copy import AE_CNN #AE_netThree_CNN
# Training
seed = 0
train_EbNodB =  8.5
val_EbNodB = train_EbNodB
training_params = [
    #batch_size, lr, ebnodb, iterations
    [1000    , 0.001,  train_EbNodB, 1000],
    [1000    , 0.0001, train_EbNodB, 10000],
    [10000   , 0.0001, train_EbNodB, 10000],
    [10000   , 0.00001, train_EbNodB, 10000]]

validation_params = [
    #batch_size, ebnodb, val_steps 
    [100000, val_EbNodB, 100],
    [100000, val_EbNodB, 1000],
    [100000, val_EbNodB, 1000],
    [100000, val_EbNodB, 1000]]

p = np.zeros([1,2,n])



model_file_CNN = 'TESTCNNnew_models_attacked_CNN/ae_k_{}_n_{}'.format(k,n)
tf.reset_default_graph()
ae_CNN = AE_CNN(k,n,seed)
ae_CNN.train(True,0, p,training_params, validation_params)
ae_CNN.save(model_file_CNN)











## train netThree_CNN and save its model
#print('training netThree_CNN started')
#model_file_netThree_CNN = 'test_models_attacked_netThree_CNN/ae_k_{}_n_{}'.format(k,n)
#ae_netThree_CNN = AE_netThree_CNN(k,n,seed)
#ae_netThree_CNN.train(True,0, p,training_params, validation_params)
#ae_netThree_CNN.save(model_file_netThree_CNN)
#print('training netThree_CNN finished and model is saved.')