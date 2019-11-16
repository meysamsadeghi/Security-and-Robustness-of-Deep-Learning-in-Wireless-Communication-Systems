# Dummy to test if it is better to SVD the normalized perturbations or if it is better to SVD the un-normalized perturbations

# Creating universal perturbation for snr 0dB
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from load_data import ModCls_loaddata
from attacks import fgm_ModCls


tf.reset_default_graph()
#==============================================================================
SNR = 0
PNR = 0
# Here we load the DATA for modulation classification
X_loaded,lbl,X_train,X_test,classes,snrs,mods,Y_train,Y_test,train_idx,test_idx = ModCls_loaddata()
# TRAINING
train_SNRs = list(map(lambda x: lbl[x][1], train_idx))
train_X_0 = X_train[np.where(np.array(train_SNRs)== SNR)].reshape(-1,2,128,1)
train_Y_0 = Y_train[np.where(np.array(train_SNRs)== SNR)] 
#TEST
test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
test_X_0 = X_test[np.where(np.array(test_SNRs)== SNR)].reshape(-1,2,128,1)
test_Y_0 = Y_test[np.where(np.array(test_SNRs)== SNR)]    

classes = mods
num_class = 11

SigPow = ( np.sum(np.linalg.norm(train_X_0.reshape([-1,256]),axis=1)) / train_X_0.shape[0] )**2 # I used to use 0.01 since np.linalg.norm(test_X_0[i]) = 0.09339~ 0.1
dim_X_vec = np.array([1,2,4]) #np.array([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]) # outer loop for cnt_dim
num_times = 2 # iner loop for rnd
aid = 10**((PNR-SNR)/10)
Epsilon_uni = np.sqrt(SigPow * aid) 


#===================================================================================================
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('ModulationCls/tfTrainedmodel.meta')                           #('ModClass/tfmodel.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('ModulationCls'))    #('ModClass'))
    graph = tf.get_default_graph() 
    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    cost = graph.get_tensor_by_name("cost:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    predictions = graph.get_tensor_by_name("predictions:0")
    grad = tf.gradients(cost,X)
    #
    acc_optimal_moosavi = 1
    acc_optimal_pca_max_un = 1
    acc_optimal_pca_all_un = 1
    acc_optimal_pca_max_nor  = 1
    acc_optimal_pca_all_nor  = 1
    #
    acc_moosavi = np.zeros(len(dim_X_vec))
    acc_pcamax_un = np.zeros(len(dim_X_vec))
    acc_pcaall_un = np.zeros(len(dim_X_vec))
    acc_pcamax_n = np.zeros(len(dim_X_vec))
    acc_pcaall_n = np.zeros(len(dim_X_vec))
    for cnt_dim in range(len(dim_X_vec)):
        N = dim_X_vec[cnt_dim]
        vec_moosavi = np.zeros([num_times,1])
        vec_pcamax_un = np.zeros([num_times,1])
        vec_pcaall_un = np.zeros([num_times,1])
        vec_pcamax_n = np.zeros([num_times,1])
        vec_pcaall_n = np.zeros([num_times,1])
        for rnd in range(num_times): # this is the number of times we randomly select N data points
            np.random.seed()
            selcted_imgs = np.random.choice(range(train_X_0.shape[0]), size= N, replace=False)
            #========================================================
            # Moosavi's method for creating universal perturbation
            delta = 0.1
            Error = 0
            universal_per = np.zeros([2,128,1])
            cnr = 0
            while (Error < (1-delta)) and (cnr<2000):
                cnr = cnr +1
                true_rate = 0
                for cnr_index in range(N):#
                    input_image = train_X_0[selcted_imgs[cnr_index]].reshape([-1,2,128,1])
                
                    predicted_clean = np.argmax(sess.run(predictions, feed_dict={X:input_image, is_training: False}))
                    predicted_adv = np.argmax(sess.run(predictions, feed_dict={X: (input_image + universal_per.reshape([1,2,128,1])), is_training: False}))
                    if predicted_adv == predicted_clean:#predicted_label == np.argmax(train_Y_0[cnr_index]):
                        true_rate = true_rate + 1
                        # First we need to find adverssarial direction for this instant  by solving (1), or using fgm or deepfool
                        _, adv_perturbation, _, _ = fgm_ModCls((input_image + universal_per.reshape([1,2,128,1])), np.eye(num_class)[predicted_clean,:], num_class)
                        adv_perturbn_reshaped = adv_perturbation.reshape([2,128,1])
                        # Second we need to revise the universal perturbation
                        if np.linalg.norm(universal_per + adv_perturbn_reshaped) < Epsilon_uni: 
                            universal_per = universal_per + adv_perturbn_reshaped
                        else:
                            universal_per =  Epsilon_uni * ((universal_per + adv_perturbn_reshaped)/(np.linalg.norm(universal_per + adv_perturbn_reshaped)))
                
                Error = 1-(true_rate/N)
            universal_per = Epsilon_uni * (1 / np.linalg.norm(universal_per) ) * universal_per
            #========================================================
            # un-normalized PCA for creating universal perturbations
            adv_dir_matrix = np.zeros([N,256])
            adv_dir_matrix_normlzd= np.zeros([N,256])
            #norm_adv_dir_matrix = np.zeros([dim_X_v,1])
            for sam_index in range(N):
                input_image = train_X_0[selcted_imgs[sam_index]].reshape([-1,2,128,1])
                _, adv_perturbation, _, _ = fgm_ModCls(input_image, train_Y_0[sam_index], num_class)
                adv_dir_matrix[sam_index,:] = adv_perturbation.reshape([256]) # it should be [256,1] I changed it and dont know if it is correct, check
                #norm_adv_dir_matrix[sam_index,0] = np.linalg.norm(adv_perturbation)
                adv_dir_matrix_normlzd[sam_index,:] = (1 / (np.linalg.norm(adv_perturbation)+ 0.000000000001)) * adv_perturbation.reshape([256])
            # PCA un-normalized
            U_mat,Sigma,V_matrix_Transpose = np.linalg.svd(adv_dir_matrix)
            V_matrix = V_matrix_Transpose.T
            # Universal perturbation based on maximum pricipal direction
            uni_max_sv = (Epsilon_uni / np.linalg.norm(V_matrix[:,0])) * (V_matrix[:,0])
            # UNiversal perturbation based on all principal directions
            uni_all_sv = np.zeros([256,1])
            for cnr in range(len(Sigma)):
                uni_all_sv = uni_all_sv + ( (Sigma[cnr]**2 / np.sum(Sigma**2)) * V_matrix[:,cnr] ).reshape([256,1])
            uni_all_sv =  (Epsilon_uni / np.linalg.norm(uni_all_sv)) * uni_all_sv
            # normalized PCA for creating universal perturbations
            U_mat,Sigma_normlzd,V_matrix_Transpose_normlzd = np.linalg.svd(adv_dir_matrix_normlzd)
            V_matrix_normlzd = V_matrix_Transpose_normlzd.T
            # Universal perturbation based on maximum pricipal direction
            uni_max_sv_normlzd = (Epsilon_uni / np.linalg.norm(V_matrix_normlzd[:,0])) * (V_matrix_normlzd[:,0])
            #
            uni_all_sv_normlzd = np.zeros([256,1])
            for cnr in range(len(Sigma_normlzd)):
                uni_all_sv_normlzd = uni_all_sv_normlzd + ( (Sigma_normlzd[cnr]**2 / np.sum(Sigma_normlzd**2)) * V_matrix_normlzd[:,cnr] ).reshape([256,1])
            uni_all_sv_normlzd =  (Epsilon_uni / np.linalg.norm(uni_all_sv_normlzd)) * uni_all_sv_normlzd
            #========================================================
            accuracy_clean = 0
            accuracy_attacked = 0
            accuracy_pca_max = 0
            accuracy_pca_all = 0
            accuracy_pca_max_nrmlzd = 0
            accuracy_pca_all_nrmlzd = 0
            num_samples = test_X_0.shape[0]
            
            for i in range(num_samples):
                input_image = test_X_0[i].reshape([-1,2,128,1])
                pred_clean = np.argmax(sess.run(predictions, feed_dict={X: input_image, is_training: False}))
                pred_attacked = np.argmax(sess.run(predictions, feed_dict={X: (input_image + universal_per.reshape([1,2,128,1])), is_training: False}))
                pred_pca_max = np.argmax(sess.run(predictions, feed_dict={X: (input_image + uni_max_sv.reshape([1,2,128,1])), is_training: False}))
                pred_pca_all = np.argmax(sess.run(predictions, feed_dict={X: (input_image + uni_all_sv.reshape([1,2,128,1])), is_training: False}))
                pred_pca_max_nrmlzd = np.argmax(sess.run(predictions, feed_dict={X: (input_image + uni_max_sv_normlzd.reshape([1,2,128,1])), is_training: False}))
                pred_pca_all_nrmlzd = np.argmax(sess.run(predictions, feed_dict={X: (input_image + uni_all_sv_normlzd.reshape([1,2,128,1])), is_training: False}))
                
                
                if np.argmax(test_Y_0[i]) == pred_clean:
                    accuracy_clean = accuracy_clean + 1
                if np.argmax(test_Y_0[i]) == pred_attacked:
                    accuracy_attacked = accuracy_attacked + 1
                if np.argmax(test_Y_0[i]) == pred_pca_max:
                    accuracy_pca_max = accuracy_pca_max + 1
                if np.argmax(test_Y_0[i]) == pred_pca_all:
                    accuracy_pca_all = accuracy_pca_all + 1
                if np.argmax(test_Y_0[i]) == pred_pca_max_nrmlzd:
                    accuracy_pca_max_nrmlzd = accuracy_pca_max_nrmlzd + 1
                if np.argmax(test_Y_0[i]) == pred_pca_all_nrmlzd:
                    accuracy_pca_all_nrmlzd = accuracy_pca_all_nrmlzd + 1
            
            
            acc_attacked = accuracy_attacked / num_samples
            if acc_optimal_moosavi > acc_attacked:
                acc_optimal_moosavi = acc_attacked
                optimal_moosavi = universal_per.reshape([256,1])
                optimal_sample_number_moosavi = dim_X_vec[cnt_dim]
            
            
            acc_pca_max = accuracy_pca_max / num_samples
            if acc_optimal_pca_max_un > acc_pca_max:
                acc_optimal_pca_max_un = acc_pca_max
                optimal_pca_max_un = uni_max_sv.reshape([256,1])
                optimal_sample_number_pcamax_un = dim_X_vec[cnt_dim]
            
            
            acc_pca_all = accuracy_pca_all / num_samples
            if acc_optimal_pca_all_un > acc_pca_all:
                acc_optimal_pca_all_un = acc_pca_all
                optimal_pca_all_un = uni_all_sv.reshape([256,1])
                optimal_sample_number_pcaall_un = dim_X_vec[cnt_dim]
            
            acc_pca_max_nrmlzd = accuracy_pca_max_nrmlzd / num_samples
            if acc_optimal_pca_max_nor > acc_pca_max_nrmlzd:
                acc_optimal_pca_max_nor = acc_pca_max_nrmlzd
                optimal_pca_max_nor = uni_max_sv_normlzd.reshape([256,1])
                optimal_sample_number_pcamax = dim_X_vec[cnt_dim]
                
            
            acc_pca_all_nrmlzd = accuracy_pca_all_nrmlzd / num_samples
            if acc_optimal_pca_all_nor > acc_pca_all_nrmlzd:
                acc_optimal_pca_all_nor = acc_pca_all_nrmlzd
                optimal_pca_all_nor = uni_all_sv_normlzd.reshape([256,1])
                optimal_sample_number_pcaall = dim_X_vec[cnt_dim]
            
            # Now sum over the for
            vec_moosavi[rnd] = acc_attacked
            vec_pcamax_un[rnd] = acc_pca_max
            vec_pcaall_un[rnd] =  acc_pca_all
            vec_pcamax_n[rnd] =  acc_pca_max_nrmlzd
            vec_pcaall_n[rnd] =  acc_pca_all_nrmlzd
            
        # Now average to see waht is the real performance
        print('N', N)
        
        
        acc_moosavi[cnt_dim] = np.sum(vec_moosavi) / num_times
        print('moosavi acc', acc_moosavi[cnt_dim] )
        
        acc_pcamax_un[cnt_dim]  = np.sum(vec_pcamax_un) / num_times
        print('acc_pcamax_un acc', acc_pcamax_un[cnt_dim] )
        
        acc_pcaall_un[cnt_dim]  = np.sum(vec_pcaall_un) / num_times
        print('acc_pcaall_un acc', acc_pcaall_un[cnt_dim] )
        
        acc_pcamax_n[cnt_dim]  = np.sum(vec_pcamax_n) / num_times
        print('acc_pcamax_n acc', acc_pcamax_n[cnt_dim] )
        
        acc_pcaall_n[cnt_dim]  = np.sum(vec_pcaall_n) / num_times
        print('acc_pcaall_n acc', acc_pcaall_n[cnt_dim] )
        

import pickle
pickle.dump([acc_moosavi,acc_pcamax_un,acc_pcaall_un,acc_pcamax_n,acc_pcaall_n],open('SNR10_PNR0','wb'))  
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
