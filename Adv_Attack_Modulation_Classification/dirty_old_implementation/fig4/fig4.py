# fig4 - Illustrating Transferability and shift invariance properties of UAPs
# 1. import the UAP for SNR=10 and PSR==-10, call it UAP_wb
# 2. import the VT-CNN2
# 3. import the UAP of MLP and call it UAP_bb  ==> this uses fig4MLP.py which generate UAP_bb for the MLP and then save the best
# 4. attack VT-CNN2 with shifted versions of UAP_wb and UAP_bb
# 5. plot the comparison in a bar figure




import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/meysam/ML/ModulationClassification/codes_letter')
from load_data import ModCls_loaddata



tf.reset_default_graph()
#===================  Initilaization of mainparameters ==================
PSR = -10
PNR = SNR + PSR
N = 50
SNR=10
#===================  Import the Data ==================
# Here we load the DATA for modulation classification
X_loaded,lbl,X_train,X_test,classes,snrs,mods,Y_train,Y_test,train_idx,test_idx = ModCls_loaddata()
# TRAINING
train_SNRs = list(map(lambda x: lbl[x][1], train_idx))
train_X_0 = X_train[np.where(np.array(train_SNRs)==SNR)].reshape(-1,2,128,1)
train_Y_0 = Y_train[np.where(np.array(train_SNRs)==SNR)] 
#TEST
SNR = 10
test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
test_X_10 = X_test[np.where(np.array(test_SNRs)==SNR)].reshape(-1,2,128,1)
test_Y_10 = Y_test[np.where(np.array(test_SNRs)==SNR)]    

SNR = 0
test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
test_X_0 = X_test[np.where(np.array(test_SNRs)==SNR)].reshape(-1,2,128,1)
test_Y_0 = Y_test[np.where(np.array(test_SNRs)==SNR)]  

classes = mods
num_class = 11

#===================  Initialize ==================
SigPow = ( np.sum(np.linalg.norm(train_X_0.reshape([-1,256]),axis=1)) / train_X_0.shape[0] )**2 # SigPow is roughly the same for SNR=0 and SNR=10
aid = 10**(PSR/10)
Epsilon_uni = np.sqrt(SigPow * aid) 
max_num_iter_moosavi = 1


#=================== Import UAPs for MLP (black box attack) and white box attack ==================
import pickle

with open('/home/meysam/ML/ModulationClassification/codes_letter/fig3/PSR_SNR10_complete.pkl', 'rb') as f:
    [_,_,_,_,_,_,Grad_Opt] = pickle.load(f)
WB_SNR10 = Grad_Opt[:,10] # for PSR=-10
WB_SNR0  = Grad_Opt[:,20]


with open('MLP_UAP_SNR0_PSR-10.pkl', 'rb') as f: # 38% BEST UP TO HERE # this one generates UAP for SNR=0 and then apply it on SNR=+10
    grad_per_n = pickle.load(f).reshape([1,2,128,1])
BB_10= ((Epsilon_uni / np.linalg.norm(grad_per_n)) * grad_per_n).reshape([1,2,128,1])
BB_0 = ((np.linalg.norm(WB_SNR0)/ np.linalg.norm(grad_per_n)) * grad_per_n).reshape([1,2,128,1])

#================================== Rolling =========================================================
rolled_WB10_100 = np.roll(WB_SNR10.reshape(2,128),100,axis=1).reshape(1,2,128,1)
rolled_WB10_80 = np.roll(WB_SNR10.reshape(2,128),80,axis=1).reshape(1,2,128,1)
rolled_WB10_60 = np.roll(WB_SNR10.reshape(2,128),60,axis=1).reshape(1,2,128,1)
rolled_WB10_40 = np.roll(WB_SNR10.reshape(2,128),40,axis=1).reshape(1,2,128,1)
rolled_WB10_20 = np.roll(WB_SNR10.reshape(2,128),20,axis=1).reshape(1,2,128,1)

rolled_WB0_100 = np.roll(WB_SNR0.reshape(2,128),100,axis=1).reshape(1,2,128,1)
rolled_WB0_80 = np.roll(WB_SNR0.reshape(2,128),80,axis=1).reshape(1,2,128,1)
rolled_WB0_60 = np.roll(WB_SNR0.reshape(2,128),60,axis=1).reshape(1,2,128,1)
rolled_WB0_40 = np.roll(WB_SNR0.reshape(2,128),40,axis=1).reshape(1,2,128,1)
rolled_WB0_20 = np.roll(WB_SNR0.reshape(2,128),20,axis=1).reshape(1,2,128,1)

rolled_BB10_100 = np.roll(BB_10.reshape(2,128),100,axis=1).reshape(1,2,128,1)
rolled_BB10_80 = np.roll(BB_10.reshape(2,128),80,axis=1).reshape(1,2,128,1)
rolled_BB10_60 = np.roll(BB_10.reshape(2,128),60,axis=1).reshape(1,2,128,1)
rolled_BB10_40 = np.roll(BB_10.reshape(2,128),40,axis=1).reshape(1,2,128,1)
rolled_BB10_20 = np.roll(BB_10.reshape(2,128),20,axis=1).reshape(1,2,128,1)

rolled_BB0_100 = np.roll(BB_0.reshape(2,128),100,axis=1).reshape(1,2,128,1)
rolled_BB0_80 = np.roll(BB_0.reshape(2,128),80,axis=1).reshape(1,2,128,1)
rolled_BB0_60 = np.roll(BB_0.reshape(2,128),60,axis=1).reshape(1,2,128,1)
rolled_BB0_40 = np.roll(BB_0.reshape(2,128),40,axis=1).reshape(1,2,128,1)
rolled_BB0_20 = np.roll(BB_0.reshape(2,128),20,axis=1).reshape(1,2,128,1)
#===================================================================================================

with tf.Session() as sess:
    # Import the MLP graph
    new_saver = tf.train.import_meta_graph('/home/meysam/ML/ModulationClassification/codes_letter/ModulationCls/tfTrainedmodel.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('/home/meysam/ML/ModulationClassification/codes_letter/ModulationCls/'))    #('ModClass'))
    graph = tf.get_default_graph() 
    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    cost = graph.get_tensor_by_name("cost:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    predictions = graph.get_tensor_by_name("predictions:0")
    grad = tf.gradients(cost,X)

    # =============== SNR=10 ======================
    accuracy_clean_10  = 0
    
    accuracy_WB_10 = 0
    accuracy_WB_10_roll20 = 0
    accuracy_WB_10_roll40 = 0
    accuracy_WB_10_roll60 = 0
    accuracy_WB_10_roll80 = 0
    accuracy_WB_10_roll100 = 0
    
    accuracy_BB_10 = 0
    accuracy_BB_10_roll20 = 0
    accuracy_BB_10_roll40 = 0
    accuracy_BB_10_roll60 = 0
    accuracy_BB_10_roll80 = 0
    accuracy_BB_10_roll100 = 0
    
    
    num_samples = test_X_10.shape[0]
    for i in range(num_samples):
        input_image = test_X_10[i].reshape([-1,2,128,1])
        pred_clean = np.argmax(sess.run(predictions, feed_dict={X: input_image, is_training: False}))
        
        pred_WB = np.argmax(sess.run(predictions, feed_dict={X: (input_image + WB_SNR10.reshape([-1,2,128,1])), is_training: False}))
        pred_WB_20 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_WB10_20), is_training: False}))
        pred_WB_40 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_WB10_40), is_training: False}))
        pred_WB_60 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_WB10_60), is_training: False}))
        pred_WB_80 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_WB10_80), is_training: False}))
        pred_WB_100 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_WB10_100), is_training: False}))
        
        
        pred_grad_n = np.argmax(sess.run(predictions, feed_dict={X: (input_image + BB_10), is_training: False}))
        pred_grad_n_20 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_BB10_20), is_training: False}))
        pred_grad_n_40 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_BB10_40), is_training: False}))
        pred_grad_n_60 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_BB10_60), is_training: False}))
        pred_grad_n_80 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_BB10_80), is_training: False}))
        pred_grad_n_100 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_BB10_100), is_training: False}))
        
        if np.argmax(test_Y_10[i]) == pred_clean:
            accuracy_clean_10 = accuracy_clean_10 + (1/ num_samples)
            
        if np.argmax(test_Y_10[i]) == pred_WB:
            accuracy_WB_10 = accuracy_WB_10 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_WB_20:
            accuracy_WB_10_roll20 = accuracy_WB_10_roll20 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_WB_40:
            accuracy_WB_10_roll40 = accuracy_WB_10_roll40 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_WB_60:
            accuracy_WB_10_roll60 = accuracy_WB_10_roll60 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_WB_80:
            accuracy_WB_10_roll80 = accuracy_WB_10_roll80 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_WB_100:
            accuracy_WB_10_roll100 = accuracy_WB_10_roll100 + (1/ num_samples)   
        
        if np.argmax(test_Y_10[i]) == pred_grad_n:
            accuracy_BB_10 = accuracy_BB_10 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_grad_n_20:
            accuracy_BB_10_roll20 = accuracy_BB_10_roll20 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_grad_n_40:
            accuracy_BB_10_roll40 = accuracy_BB_10_roll40 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_grad_n_60:
            accuracy_BB_10_roll60 = accuracy_BB_10_roll60 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_grad_n_80:
            accuracy_BB_10_roll80 = accuracy_BB_10_roll80 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_grad_n_100:
            accuracy_BB_10_roll100 = accuracy_BB_10_roll100 + (1/ num_samples)
    
    
    print('accuracy_clean_10',accuracy_clean_10)
    
    
    print('accuracy_WB_10',accuracy_WB_10)
    print('accuracy_WB_10_roll20',accuracy_WB_10_roll20)
    print('accuracy_WB_10_roll40',accuracy_WB_10_roll40)
    print('accuracy_WB_10_roll60',accuracy_WB_10_roll60)
    print('accuracy_WB_10_roll80',accuracy_WB_10_roll80)
    print('accuracy_WB_10_roll100',accuracy_WB_10_roll100)
    
    print('accuracy_BB_10',accuracy_BB_10)
    print('accuracy_BB_10_roll20',accuracy_BB_10_roll20)
    print('accuracy_BB_10_roll40',accuracy_BB_10_roll40)
    print('accuracy_BB_10_roll60',accuracy_BB_10_roll60)
    print('accuracy_BB_10_roll80',accuracy_BB_10_roll80)
    print('accuracy_BB_10_roll100',accuracy_BB_10_roll100)
    
    # =============== SNR=0 ======================
    accuracy_clean_0  = 0
    
    accuracy_WB_0 = 0
    accuracy_WB_0_roll20 = 0
    accuracy_WB_0_roll40 = 0
    accuracy_WB_0_roll60 = 0
    accuracy_WB_0_roll80 = 0
    accuracy_WB_0_roll100 = 0
    
    accuracy_BB_0 = 0
    accuracy_BB_0_roll20 = 0
    accuracy_BB_0_roll40 = 0
    accuracy_BB_0_roll60 = 0
    accuracy_BB_0_roll80 = 0
    accuracy_BB_0_roll100 = 0
    
    num_samples = test_X_0.shape[0]
    for i in range(num_samples):
        input_image = test_X_0[i].reshape([-1,2,128,1])
        pred_clean = np.argmax(sess.run(predictions, feed_dict={X: input_image, is_training: False}))
        
        pred_WB = np.argmax(sess.run(predictions, feed_dict={X: (input_image + WB_SNR0.reshape([-1,2,128,1])), is_training: False}))
        pred_WB_20 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_WB0_20), is_training: False}))
        pred_WB_40 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_WB0_40), is_training: False}))
        pred_WB_60 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_WB0_60), is_training: False}))
        pred_WB_80 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_WB0_80), is_training: False}))
        pred_WB_100 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_WB0_100), is_training: False}))
        pred_grad_n = np.argmax(sess.run(predictions, feed_dict={X: (input_image + BB_0), is_training: False}))
        
        pred_grad_n_20 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_BB0_20), is_training: False}))
        pred_grad_n_40 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_BB0_40), is_training: False}))
        pred_grad_n_60 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_BB0_60), is_training: False}))
        pred_grad_n_80 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_BB0_80), is_training: False}))
        pred_grad_n_100 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + rolled_BB0_100), is_training: False}))
        
        
        if np.argmax(test_Y_0[i]) == pred_clean:
            accuracy_clean_0 = accuracy_clean_0 + (1/ num_samples)
        
        
        if np.argmax(test_Y_0[i]) == pred_WB:
            accuracy_WB_0 = accuracy_WB_0 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_WB_20:
            accuracy_WB_0_roll20 = accuracy_WB_0_roll20 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_WB_40:
            accuracy_WB_0_roll40 = accuracy_WB_0_roll40 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_WB_60:
            accuracy_WB_0_roll60 = accuracy_WB_0_roll60 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_WB_80:
            accuracy_WB_0_roll80 = accuracy_WB_0_roll80 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_WB_100:
            accuracy_WB_0_roll100 = accuracy_WB_0_roll100 + (1/ num_samples)
        
        
        if np.argmax(test_Y_0[i]) == pred_grad_n:
            accuracy_BB_0 = accuracy_BB_0 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_grad_n_20:
            accuracy_BB_0_roll20 = accuracy_BB_0_roll20 + (1/ num_samples)
        if np.argmax(test_Y_import pickle
            accuracy_BB_0_roll40 = accuracy_BB_0_roll40 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_grad_n_60:
            accuracy_BB_0_roll60 = accuracy_BB_0_roll60 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_grad_n_80:
            accuracy_BB_0_roll80 = accuracy_BB_0_roll80 + (1/ num_samples)
        if np.argmax(test_Y_10[i]) == pred_grad_n_100:
            accuracy_BB_0_roll100 = accuracy_BB_0_roll100 + (1/ num_samples)    
            
        
            
    print('accuracy_clean_0',accuracy_clean_0)   
        
    print('accuracy_WB_0',accuracy_WB_0)
    print('accuracy_WB_0_roll20',accuracy_WB_0_roll20)
    print('accuracy_WB_0_roll40',accuracy_WB_0_roll40)
    print('accuracy_WB_0_roll60',accuracy_WB_0_roll60)
    print('accuracy_WB_0_roll80',accuracy_WB_0_roll80)
    print('accuracy_WB_0_roll100',accuracy_WB_0_roll100)
    
    print('accuracy_BB_0',accuracy_BB_0)
    print('accuracy_BB_0_roll20',accuracy_BB_0_roll20)
    print('accuracy_BB_0_roll40',accuracy_BB_0_roll40)
    print('accuracy_BB_0_roll60',accuracy_BB_0_roll60)
    print('accuracy_BB_0_roll80',accuracy_BB_0_roll80)
    print('accuracy_BB_0_roll100',accuracy_BB_0_roll100) 
        
        



#===================================================================================================
acc_WB_SNR10 = np.array([accuracy_WB_10,accuracy_WB_10_roll20,accuracy_WB_10_roll40,accuracy_WB_10_roll60,accuracy_WB_10_roll80,accuracy_WB_10_roll100])      
acc_BB_SNR10 = np.array([accuracy_BB_10,accuracy_BB_10_roll20,accuracy_BB_10_roll40,accuracy_BB_10_roll60,accuracy_BB_10_roll80,accuracy_BB_10_roll100])               
        
acc_WB_SNR0 = np.array([accuracy_WB_0,accuracy_WB_0_roll20,accuracy_WB_0_roll40,accuracy_WB_0_roll60,accuracy_WB_0_roll80,accuracy_WB_0_roll100])
acc_BB_SNR0 = np.array([accuracy_BB_0,accuracy_BB_0_roll20,accuracy_BB_0_roll40,accuracy_BB_0_roll60,accuracy_BB_0_roll80,accuracy_BB_0_roll100])       
        
Shifts = np.array([0,20,40,60,80,100])    
        
fig, ax = plt.subplots()

ax.plot(Shifts,75 * np.ones([6,1]),'k*-',label='No attack - SNR=10 dB') 
ax.plot(Shifts,71 * np.ones([6,1]),'kh--',label='No attack - SNR=0 dB') 

ax.plot(Shifts,100 * acc_BB_SNR10,'r^-',label='Black-box attack - SNR=10 dB')    
ax.plot(Shifts,100 * acc_WB_SNR10,'go-',label='White-box attack - SNR=10 dB')        

ax.plot(Shifts,100 * acc_BB_SNR0,'ms--',label='Black-box attack - SNR=0 dB')
ax.plot(Shifts,100 * acc_WB_SNR0,'bX--',label='White-box attack - SNR=0 dB') 
  

plt.legend(loc='upper right')
plt.xticks(Shifts,Shifts)

ax.set_xlabel('Number of shifted elements')
ax.set_ylabel('Accuracy')

ax.grid(True)
plt.savefig("fig4.pdf")
plt.show()        

#===================================================================================================
import pickle
with open('fig4_data','wb') as f:
    pickle.dump([acc_WB_SNR10,acc_BB_SNR10,acc_WB_SNR0,acc_BB_SNR0],f)        