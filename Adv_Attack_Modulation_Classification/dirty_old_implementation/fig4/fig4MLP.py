# This file (fig4MLP.py) generates UAP_bb for the MLP and then save the best
# 1. first it loads the MLP graph and meta data
# 2. generates UAP for SNR=10 and PSR=-10
# 3. Save the best UAP on the substitue MLP 


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/meysam/ML/ModulationClassification/codes_letter')
from load_data import ModCls_loaddata
#from attacks import fgm_ModCls



tf.reset_default_graph()
#=================== ZERO : Initilaization of mainparameters ==================
SNR = 0
PSR = -10
PNR = SNR + PSR
N = 50
Num_iter = 3000 # number of times it select 50 samles and generate a UAP, the bigger it ecomes the better UAP we can expect to find
acc_optimal_grad_n = 1 # to save the best UAP we need this
#=================== First: Import the Data ==================
# Here we load the DATA for modulation classification
X_loaded,lbl,X_train,X_test,classes,snrs,mods,Y_train,Y_test,train_idx,test_idx = ModCls_loaddata()
# TRAINING
train_SNRs = list(map(lambda x: lbl[x][1], train_idx))
train_X_0 = X_train[np.where(np.array(train_SNRs)==SNR)].reshape(-1,256)#(-1,2,128,1)
train_Y_0 = Y_train[np.where(np.array(train_SNRs)==SNR)] 
#TEST
test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
test_X_0 = X_test[np.where(np.array(test_SNRs)==SNR)].reshape(-1,256)#(-1,2,128,1)
test_Y_0 = Y_test[np.where(np.array(test_SNRs)==SNR)]    


classes = mods
num_class = 11

#=================== Second: Initialize ==================
SigPow = ( np.sum(np.linalg.norm(train_X_0.reshape([-1,256]),axis=1)) / train_X_0.shape[0] )**2 # I used to use 0.01 since np.linalg.norm(test_X_0[i]) = 0.09339~ 0.1
aid = 10**(PSR/10)
Epsilon_uni = np.sqrt(SigPow * aid) 
max_num_iter_moosavi = 1
#===================================================================================================
with tf.Session() as sess:
    # Import the MLP graph
    new_saver = tf.train.import_meta_graph('/home/meysam/ML/ModulationClassification/codes_first_version/ModulationCls_TFMeysamMLP/TF_MeysamModel_MLP.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('/home/meysam/ML/ModulationClassification/codes_first_version/ModulationCls_TFMeysamMLP/'))    #('ModClass'))
    graph = tf.get_default_graph() 
    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    cost = graph.get_tensor_by_name("cost:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    predictions = graph.get_tensor_by_name("predictions:0")
    grad = tf.gradients(cost,X)
    
    # ================ Iterate Num_iter times to find a good UAP
    for cnr in range(Num_iter):
        print('cnr:',cnr)
        # STEP.1. =============== Creata a UAP attack ====
        np.random.seed()
        selcted_imgs = np.random.choice(range(train_X_0.shape[0]), size= N, replace=False)
        grad_matrix_n = np.zeros([N,256])
        for ctr_index in range(N):
            input_image = train_X_0[selcted_imgs[ctr_index]].reshape([-1,256])#reshape([-1,2,128,1])
            temp = np.asarray(sess.run(grad,feed_dict={X: input_image, is_training: False, Y: train_Y_0[selcted_imgs[ctr_index]].reshape(1,11)})).reshape(1,256)
            grad_matrix_n[ctr_index,:] = temp / (np.linalg.norm(temp) + 0.00000001)
            
        _,_,v_n_T = np.linalg.svd(grad_matrix_n)
        grad_per_n = Epsilon_uni * (1 / np.linalg.norm(v_n_T.T[:,0])) * v_n_T.T[:,0]
        # STEP.2. =============== Test the attack performance on the substitue MLP ===== 
        accuracy_clean  = 0
        accuracy_grad_n = 0
        num_samples = test_X_0.shape[0]
        for i in range(num_samples):
            input_image = test_X_0[i].reshape([-1,256])
            #pred_clean = np.argmax(sess.run(predictions, feed_dict={X: input_image, is_training: False}))
            pred_grad_n = np.argmax(sess.run(predictions, feed_dict={X: (input_image + grad_per_n.reshape([-1,256])), is_training: False}))
    
            #if np.argmax(test_Y_0[i]) == pred_clean:
                #accuracy_clean = accuracy_clean + (1/ num_samples)
            if np.argmax(test_Y_0[i]) == pred_grad_n:
                accuracy_grad_n = accuracy_grad_n + (1/ num_samples)
        print('accuracy_grad_n',accuracy_grad_n)
        #print('accuracy_clean',accuracy_clean)
        # STEP.3. =============== select the best UAP
        if acc_optimal_grad_n > accuracy_grad_n:
            acc_optimal_grad_n = accuracy_grad_n
            optimal_grad_n = grad_per_n.reshape([256])

        
    #=====================================================================

import pickle
with open('MLP_UAP_SNR0_PSR-10_1000rounds.pkl', 'wb') as f:
    pickle.dump(grad_per_n, f)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    