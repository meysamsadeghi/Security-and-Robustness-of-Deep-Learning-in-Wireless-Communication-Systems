# ALL THE ATTACKS
import numpy as np
import tensorflow as tf




#===================================================================================================
# Adversarial Generator
def fgm_ModCls(in_img,in_label,num_class):
    ########################################
    sess = tf.get_default_session()
    graph = tf.get_default_graph()
    graph = tf.get_default_graph() 
    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    #logits = graph.get_tensor_by_name('logits:0')
    is_training = graph.get_tensor_by_name("is_training:0")
    cost = graph.get_tensor_by_name("cost:0")
    #accuracy = graph.get_tensor_by_name("accuracy:0")
    predictions = graph.get_tensor_by_name("predictions:0")
    grad = tf.gradients(cost,X, name='grad')
    ########################################
    eps_acc = 0.00001 * np.linalg.norm(in_img)
    epsilon_vector = np.zeros([num_class])
    for cls in range(num_class):
        y_target = np.eye(num_class)[cls,:].reshape(1,num_class)
        adv_per_needtoreshape = -1 * np.asarray(sess.run(grad,feed_dict={X: in_img, is_training: False, Y: y_target}))
        adv_per = adv_per_needtoreshape.reshape(1,2,128,1)
        norm_adv_per = adv_per / (np.linalg.norm(adv_per) +  0.000000000001)
        epsilon_max = 1 * np.linalg.norm(in_img)
        epsilon_min = 0
        num_iter = 0
        wcount = 0
        while (epsilon_max-epsilon_min > eps_acc) and (num_iter < 30):
            wcount = wcount+1
            num_iter = num_iter +1
            epsilon = (epsilon_max + epsilon_min)/2
            adv_img_givencls = in_img + (epsilon * norm_adv_per)
            predicted_probabilities = sess.run(predictions, feed_dict={X: adv_img_givencls, is_training: False})
            
            #print('predicted label in round',wcount, 'is:', np.argmax(predicted_probabilities))
            #print(epsilon)
            compare = np.equal(np.argmax(predicted_probabilities),np.argmax(in_label))
            if compare:
                epsilon_min = epsilon
            else:
                epsilon_max = epsilon
        #print('epsilon:', epsilon)
        epsilon_vector[cls] = epsilon + eps_acc
        #print('the adversarial prediction is:', predicted_probabilities)
        #print(adv_img_givencls[0,0,10:14])
    false_cls = np.argmin(epsilon_vector)
    minimum_epsilon = np.min(epsilon_vector)
    adv_dirc = -1 * np.asarray(sess.run(grad,feed_dict={X: in_img, is_training: False, Y: np.eye(num_class)[false_cls,:].reshape(1,num_class)})).reshape(1,2,128,1)
    norm_adv_dirc = adv_dirc / (np.linalg.norm(adv_dirc) + 0.000000000001)
    adv_perturbation = minimum_epsilon * norm_adv_dirc
    adv_image = in_img + adv_perturbation
    #print('Tue label is ', np.argmax(in_label), 'and the adversay label', np.argmax(sess.run(predictions, feed_dict={X: adv_image, is_training: False})))
    #print(minimum_epsilon)
    return adv_image, adv_perturbation, false_cls, minimum_epsilon
#====================================================================================

