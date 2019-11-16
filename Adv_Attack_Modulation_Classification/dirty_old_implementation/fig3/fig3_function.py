def acc_psr_fun(PSR,SNR,num_times,N):
    '''
    :param num_time: number of Monte Carlo runs.
    :param N: number of samples to be used.
    '''
    import sys
    sys.path.insert(0, '/home/meysam/Work/ModulationClassification/codes_letter')
    sys.path.insert(0,'/home/meysam/Work/ModulationClassification/codes_letter/ModulationCls_GPU')
    import time
    import numpy as np
    import tensorflow as tf
    from load_data import ModCls_loaddata
    from attacks import fgm_ModCls
    
    print('importing was successful')
    tf.reset_default_graph()
    #==============================================================================
    PNR = PSR + SNR
    print('PSR = ', PNR - SNR)
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
    
    num_class = 11
    
    SigPow = ( np.sum(np.linalg.norm(train_X_0.reshape([-1,256]),axis=1)) / train_X_0.shape[0] )**2 # I used to use 0.01 since np.linalg.norm(test_X_0[i]) = 0.09339~ 0.1

     
    aid = 10**((PNR-SNR)/10)
    Epsilon_uni = np.sqrt(SigPow * aid) 
    
    
    max_num_iter_moosavi = 1
    

    print('Epsilon_uni', Epsilon_uni)
    
    #
    optimal_grad_n = np.zeros([256,1])
    ###
    noise_mean = np.sum(np.mean(test_X_0.reshape(-1,256),axis=1)) / test_X_0.shape[0]
    noise_std = np.sum(np.std(test_X_0.reshape(-1,256),axis=1)) / test_X_0.shape[0]
    noise = np.random.normal(noise_mean,noise_std,[1,2,128,1])
    noise_per = Epsilon_uni * (1/np.linalg.norm(noise)) * noise
    #===================================================================================================
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('/home/meysam/Work/ModulationClassification/codes_letter/ModulationCls_GPU/tfTrainedmodel.meta')                           #('ModClass/tfmodel.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('/home/meysam/Work/ModulationClassification/codes_letter/ModulationCls_GPU'))    #('ModClass'))
        graph = tf.get_default_graph() 
        X = graph.get_tensor_by_name("X:0")
        Y = graph.get_tensor_by_name("Y:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        cost = graph.get_tensor_by_name("cost:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")
        predictions = graph.get_tensor_by_name("predictions:0")
        grad = tf.gradients(cost,X)
        #
        acc_optimal_grad_n = 1
        # 
        acc_moosavi = 0
        acc_noise = 0
        acc_grad_un = 0
        acc_grad_n = 0
        #
        time_moosavi = 0
        time_gradnorm = 0
        
        vec_moosavi = np.zeros([num_times,1])
        vec_noise = np.zeros([num_times,1])
        vec_grad_un = np.zeros([num_times,1])
        vec_grad_n  = np.zeros([num_times,1])
        for rnd in range(num_times): # this is the number of times we randomly select N data points
            print('Montecarlo in round:', rnd+1, 'out of ',num_times)
            print('Montecarlo in round:', rnd+1, 'out of ',num_times)
            print('Montecarlo in round:', rnd+1, 'out of ',num_times)
            print('Montecarlo in round:', rnd+1, 'out of ',num_times)
            print('Montecarlo in round:', rnd+1, 'out of ',num_times)
            np.random.seed()
            selcted_imgs = np.random.choice(range(train_X_0.shape[0]), size= N, replace=False)
            #========================================================
            # Moosavi's method for creating universal perturbation
            moosavi_start = time.clock()
            delta = 0.3
            Error = 0
            universal_per = np.zeros([2,128,1])
            cnr = 0
            while (Error < (1-delta)) and (cnr < max_num_iter_moosavi):
                cnr = cnr +1
                #print('rnd:', rnd+1, 'cnr:',cnr)
                true_rate = 0
                each_index_time_start = time.clock()
                for cnr_index in range(N):#
                    print('cnr_index',cnr_index)
                    inside_loop_start = time.clock()
                    input_image = train_X_0[selcted_imgs[cnr_index]].reshape([-1,2,128,1])
                    predicted_clean = np.argmax(sess.run(predictions, feed_dict={X:input_image, is_training: False}))
                    predicted_adv = np.argmax(sess.run(predictions, feed_dict={X: (input_image + universal_per.reshape([1,2,128,1])), is_training: False}))
                    if predicted_adv == predicted_clean:#predicted_label == np.argmax(train_Y_0[cnr_index]):
                        true_rate = true_rate + 1
                        # First we need to find adverssarial direction for this instant  by solving (1), or using fgm or deepfool
                        fgm_start_time = time.clock()
                        _, adv_perturbation, _, _ = fgm_ModCls((input_image + universal_per.reshape([1,2,128,1])), np.eye(num_class)[predicted_clean,:], num_class)
                        fgm_time = time.clock() - fgm_start_time
                        print('fgm_time',fgm_time)
                        adv_perturbn_reshaped = adv_perturbation.reshape([2,128,1])
                        # Second we need to revise the universal perturbation
                        if np.linalg.norm(universal_per + adv_perturbn_reshaped) < Epsilon_uni: 
                            universal_per = universal_per + adv_perturbn_reshaped
                        else:
                            universal_per =  Epsilon_uni * ((universal_per + adv_perturbn_reshaped)/(np.linalg.norm(universal_per + adv_perturbn_reshaped)))
                    print('one round of loop time',time.clock() - inside_loop_start)
                each_index_time_end = time.clock()
                print('index',cnr,'time:', each_index_time_end -each_index_time_start) 
                Error = 1-(true_rate/N)
            universal_per = Epsilon_uni * (1 / np.linalg.norm(universal_per) ) * universal_per
            moosavi_finish = time.clock()
            time_moosavi = moosavi_finish - moosavi_start
            print("time_moosavi", time_moosavi)
            #========================================================
            # Just Gradient of tageted class
            gard_start = time.clock()
            grad_matrix_un = np.zeros([N,256])
            grad_matrix_n = np.zeros([N,256])
            
            for ctr_index in range(N):
                input_image = train_X_0[selcted_imgs[ctr_index]].reshape([-1,2,128,1])
                temp = np.asarray(sess.run(grad,feed_dict={X: input_image, is_training: False, Y: train_Y_0[selcted_imgs[ctr_index]].reshape(1,11)})).reshape(1,256)
                grad_matrix_un[ctr_index,:] = temp
                grad_matrix_n[ctr_index,:] = temp / (np.linalg.norm(temp) + 0.00000001)
            
            
            _,_,v_n_T = np.linalg.svd(grad_matrix_n)
            grad_per_n = Epsilon_uni * (1 / np.linalg.norm(v_n_T.T[:,0])) * v_n_T.T[:,0]
            gard_finish = time.clock()
            time_gradnorm = gard_finish - gard_start
            
            _,_,v_un_T = np.linalg.svd(grad_matrix_un)
            grad_per_un = Epsilon_uni * (1 / np.linalg.norm(v_un_T.T[:,0])) * v_un_T.T[:,0]           
            print("time_gradnorm", time_gradnorm)
            #========================================================
            time_evaluation_start = time.clock()
            accuracy_attacked = 0
            accuracy_noise = 0
            accuracy_grad_u = 0
            accuracy_grad_n= 0
            
            num_samples = test_X_0.shape[0]
            for i in range(num_samples):
                input_image = test_X_0[i].reshape([-1,2,128,1])
                pred_attacked = np.argmax(sess.run(predictions, feed_dict={X: (input_image + universal_per.reshape([1,2,128,1])), is_training: False}))
                pred_noise = np.argmax(sess.run(predictions, feed_dict={X: input_image + noise_per, is_training: False}))
                pred_grad_un = np.argmax(sess.run(predictions, feed_dict={X: (input_image + grad_per_un.reshape([1,2,128,1])), is_training: False}))
                pred_grad_n = np.argmax(sess.run(predictions, feed_dict={X: (input_image + grad_per_n.reshape([1,2,128,1])), is_training: False}))

                if np.argmax(test_Y_0[i]) == pred_attacked:
                    accuracy_attacked = accuracy_attacked + 1
                if np.argmax(test_Y_0[i]) == pred_noise:
                    accuracy_noise = accuracy_noise + 1
                if np.argmax(test_Y_0[i]) == pred_grad_un:
                    accuracy_grad_u = accuracy_grad_u + 1
                if np.argmax(test_Y_0[i]) == pred_grad_n:
                    accuracy_grad_n = accuracy_grad_n + 1
            time_evaluation_end = time.clock()
            time_evaluation = time_evaluation_end - time_evaluation_start
            print('time_evaluation:', time_evaluation)
            #==============================================================================
            acc_attacked = accuracy_attacked / num_samples
            acc_nois = accuracy_noise / num_samples
            acc_gr_un = accuracy_grad_u / num_samples
            acc_gr_n = accuracy_grad_n / num_samples
            if acc_optimal_grad_n > acc_gr_n:
                acc_optimal_grad_n = acc_gr_n
                optimal_grad_n = grad_per_n.reshape([256])

                
            #==============================================================================
            # Now sum over the for
            vec_moosavi[rnd] = acc_attacked
            vec_noise[rnd] =  acc_nois
            vec_grad_un[rnd] =  acc_gr_un
            vec_grad_n[rnd] =  acc_gr_n

#==============================================================================
#             print('acc_attacked',acc_attacked)
#             print('acc_nois',acc_nois)
#             print('acc_gr_un',acc_gr_un)
#             print('acc_gr_n',acc_gr_n)
#==============================================================================
        
        
        # Now average to see waht is the real performance
        acc_moosavi = np.sum(vec_moosavi) / num_times
        print('moosavi acc', acc_moosavi )
        
        acc_noise  = np.sum(vec_noise) / num_times
        print('noise acc', acc_noise )
    
        acc_grad_un  = np.sum(vec_grad_un) / num_times
        print('grad_un acc', acc_grad_un )
        
        acc_grad_n  = np.sum(vec_grad_n) / num_times
        print('grad_n acc', acc_grad_n )
        
    return acc_moosavi,acc_noise,acc_grad_un,acc_grad_n,time_moosavi,time_gradnorm,optimal_grad_n, acc_optimal_grad_n


    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
