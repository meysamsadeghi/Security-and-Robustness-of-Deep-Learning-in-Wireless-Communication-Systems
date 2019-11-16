#fig 4 = with random roll
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/meysam/ML/ModulationClassification/codes_letter')
from load_data import ModCls_loaddata
import pickle


tf.reset_default_graph()





#===================  Import the Data ==================
# Here we load the DATA for modulation classification
X_loaded,lbl,X_train,X_test,classes,snrs,mods,Y_train,Y_test,train_idx,test_idx = ModCls_loaddata()
# TRAINING
train_SNRs = list(map(lambda x: lbl[x][1], train_idx))
#==============================================================================
train_X_0 = X_train[np.where(np.array(train_SNRs)==SNR)].reshape(-1,2,128,1)
# train_Y_0 = Y_train[np.where(np.array(train_SNRs)==SNR)] 
#==============================================================================
#TEST
SNR = 10
test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
#==============================================================================
# test_X_10 = X_test[np.where(np.array(test_SNRs)==SNR)].reshape(-1,2,128,1)
# test_Y_10 = Y_test[np.where(np.array(test_SNRs)==SNR)]    
#==============================================================================


test_SNRs = list(map(lambda x: lbl[x][1], test_idx))


classes = mods
num_class = 11
#===================  INITIALIZE ==================
#max_num_iter_moosavi = 1
N=50
SNR = 10
PSRvec = np.arange(-20,-9,1)

#=================== Import UAPs for MLP (black box attack) and white box attack ==================
with open('/home/meysam/ML/ModulationClassification/codes_letter/fig3/PSR_SNR10_complete.pkl', 'rb') as f:
    [_,_,_,_,_,_,Grad_Opt] = pickle.load(f)
WB_SNR10 = Grad_Opt[:,10] # for PSR=-10
WB_SNR0  = Grad_Opt[:,20]
#
with open('MLP_UAP_SNR0_PSR-10.pkl', 'rb') as f: # 38% BEST UP TO HERE # this one generates UAP for SNR=0 and then apply it on SNR=+10
    grad_per_n = pickle.load(f).reshape([1,2,128,1])

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
    #===================================================================================================
    
    #======================== SNR=10 =========================
    SNR = 10
    ACC_WB_10 = np.zeros([len(PSRvec),1])
    ACC_BB_10 = np.zeros([len(PSRvec),1])
    for psr_cnr in range(len(PSRvec)):
        PSR = PSRvec[psr_cnr]
        PNR = SNR + PSR
        train_X = X_train[np.where(np.array(train_SNRs)==SNR)].reshape(-1,2,128,1)
        SigPow = ( np.sum(np.linalg.norm(train_X.reshape([-1,256]),axis=1)) / train_X.shape[0] )**2 # SigPow is roughly the same for SNR=0 and SNR=10
        aid = 10**(PSR/10)
        Epsilon_uni = np.sqrt(SigPow * aid) 
        # create the white box and black box UAPs 
        WB_UAP_10 = Grad_Opt[:,psr_cnr] # -20,-19,-18,....-10
        BB_UAP_10 = ((Epsilon_uni / np.linalg.norm(grad_per_n)) * grad_per_n).reshape([1,2,128,1])
        rndm_shift = np.random.randint(0,128)
        BB_UAP_rndm_shift_10 = np.roll(BB_UAP_10.reshape(2,128),rndm_shift,axis=1).reshape(1,2,128,1)
        # Set the test data
        test_X = X_test[np.where(np.array(test_SNRs)==SNR)].reshape(-1,2,128,1)
        test_Y = Y_test[np.where(np.array(test_SNRs)==SNR)]  
        # 
        num_samples = test_X.shape[0]
        accuracy_WB_10 = 0
        accuracy_BB_10 = 0
        for i in range(num_samples):
            input_image = test_X[i].reshape([-1,2,128,1])
            #pred_clean = np.argmax(sess.run(predictions, feed_dict={X: input_image, is_training: False}))
            pred_WB_10 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + WB_UAP_10.reshape([-1,2,128,1])), is_training: False}))
            pred_BB_10 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + BB_UAP_rndm_shift_10), is_training: False}))
                
            if np.argmax(test_Y[i]) == pred_WB_10:
                accuracy_WB_10 = accuracy_WB_10 + (1/ num_samples)
            if np.argmax(test_Y[i]) == pred_BB_10:
                accuracy_BB_10 = accuracy_BB_10 + (1/ num_samples)
        print('PSR =', PSR)
        ACC_WB_10[psr_cnr] = accuracy_WB_10
        print('accuracy_WB',accuracy_WB_10)
        ACC_BB_10[psr_cnr] = accuracy_BB_10
        print('accuracy_BB',accuracy_BB_10)
    #======================== SNR=0 =========================
    SNR = 0
    ACC_WB_0 = np.zeros([len(PSRvec),1])
    ACC_BB_0 = np.zeros([len(PSRvec),1])
    for psr_cnr in range(len(PSRvec)):
        PSR = PSRvec[psr_cnr]
        PNR = SNR + PSR
        train_X = X_train[np.where(np.array(train_SNRs)==SNR)].reshape(-1,2,128,1)
        SigPow = ( np.sum(np.linalg.norm(train_X.reshape([-1,256]),axis=1)) / train_X.shape[0] )**2 # SigPow is roughly the same for SNR=0 and SNR=10
        aid = 10**(PSR/10)
        Epsilon_uni = np.sqrt(SigPow * aid) 
        # create the white box and black box UAPs 
        WB_UAP_0 = Grad_Opt[:,psr_cnr] # -20,-19,-18,....-10
        BB_UAP_0 = ((Epsilon_uni / np.linalg.norm(grad_per_n)) * grad_per_n).reshape([1,2,128,1])
        rndm_shift = np.random.randint(0,128)
        BB_UAP_rndm_shift_0 = np.roll(BB_UAP_0.reshape(2,128),rndm_shift,axis=1).reshape(1,2,128,1)
        # Set the test data
        test_X = X_test[np.where(np.array(test_SNRs)==SNR)].reshape(-1,2,128,1)
        test_Y = Y_test[np.where(np.array(test_SNRs)==SNR)]  
        # 
        num_samples = test_X.shape[0]
        accuracy_WB_0 = 0
        accuracy_BB_0 = 0
        for i in range(num_samples):
            input_image = test_X[i].reshape([-1,2,128,1])
            #pred_clean = np.argmax(sess.run(predictions, feed_dict={X: input_image, is_training: False}))
            pred_WB_0 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + WB_UAP_0.reshape([-1,2,128,1])), is_training: False}))
            pred_BB_0 = np.argmax(sess.run(predictions, feed_dict={X: (input_image + BB_UAP_rndm_shift_0), is_training: False}))
                
            if np.argmax(test_Y[i]) == pred_WB_0:
                accuracy_WB_0 = accuracy_WB_0 + (1/ num_samples)
            if np.argmax(test_Y[i]) == pred_BB_0:
                accuracy_BB_0 = accuracy_BB_0 + (1/ num_samples)
        print('PSR =', PSR)
        ACC_WB_0[psr_cnr] = accuracy_WB_0
        print('accuracy_WB',accuracy_WB_0)
        ACC_BB_0[psr_cnr] = accuracy_BB_0
        print('accuracy_BB',accuracy_BB_0)  




#=================================================================================================== 
[acc_moosavi_10,acc_noise_10,acc_grad_un_10,acc_grad_n_10,time_moosavi_10,time_gradnorm_10,Grad_Opt_10] = pickle.load(open('/home/meysam/ML/ModulationClassification/codes_letter/fig3/PSR_SNR10_complete.pkl','rb'))
[acc_moosavi_0,acc_noise_0,acc_grad_un_0,acc_grad_n_0,time_moosavi_0,time_gradnorm_0,Grad_Opt_0] = pickle.load(open('/home/meysam/ML/ModulationClassification/codes_letter/fig3/PSR_SNR0_complete.pkl','rb'))

#===================================================================================================    
fig, ax = plt.subplots()

ax.plot(PSRvec,75 * np.ones([11,1]),'k*-',label='No attack - SNR=10 dB')  
#ax.plot(PSRvec,71 * np.ones([11,1]),'c+--',label='No attack - SNR=0 dB')


#ax.plot(PSRvec,100 * ACC_WB_10,'r^-',label='White-box attack - SNR=10 dB')    
ax.plot(PSRvec,100 * ACC_BB_10,'rs-',label='Black-box atack with random shifs - SNR=10 dB')        
ax.plot(PSRvec,100 * acc_grad_n_10[:11],'b>-',label='White-box attack of Alg. 2 - SNR=10 dB')


#ax.plot(PSRvec,100 * ACC_WB_0,'r^-',label='White-box attack - SNR=0 dB')    
#ax.plot(PSRvec,100 * ACC_BB_0,'rs--',label='Black-box with random shifs - SNR=0 dB') 
#ax.plot(PSRvec,100 * acc_grad_n_0[:11],'m^--',label='White-box attack of Alg. 2 - SNR=0 dB')
#==============================================================================
# ax.plot(PSRvec,100 * acc_BB_SNR0,'ms--',label='Black-box attack - SNR=0 dB')
# ax.plot(PSRvec,100 * acc_WB_SNR0,'bX--',label='White-box attack - SNR=0 dB') 
#==============================================================================
plt.legend(loc='lower left')
plt.xticks(PSRvec,PSRvec)

ax.set_xlabel('PSR [dB]')
ax.set_ylabel('Accuracy')

ax.grid(True)
plt.savefig("fig4.pdf")
plt.show()        

#===================================================================================================
import pickle
with open('fig4_data','wb') as f:
    pickle.dump([PSRvec,ACC_BB_10,acc_grad_n_10,ACC_BB_0,acc_grad_n_0],f)        
