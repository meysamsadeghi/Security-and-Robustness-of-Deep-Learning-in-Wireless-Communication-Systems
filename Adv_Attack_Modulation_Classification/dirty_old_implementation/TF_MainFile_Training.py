# TF implementation of Mod. Classififcation
import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as cPickle
import time

tf.reset_default_graph()
#==============================================================================
# There is a Pickle incompatibility of numpy arrays between Python 2 and 3
# which generates ascii encoding error, to work around that we use the following instead of
# Xd = cPickle.load(open("RML2016.10a_dict.dat",'rb'))
with open('RML2016.10a_dict.dat','rb') as ff:
    u = cPickle._Unpickler( ff )
    u.encoding = 'latin1'
    Xd = u.load()

snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

# Partition the data
#  into training and test sets of the form we can train/test on 
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = int(n_examples * 0.5)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]



def to_onehot(yin):
    yy = list(yin) # This is a workaround as the map output for python3 is not a list
    yy1 = np.zeros([len(list(yy)), max(yy)+1])
    yy1[np.arange(len(list(yy))),yy] = 1
    return yy1
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))



in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods
X_train_reshaped = X_train.reshape(-1,2,128,1)
X_test_reshaped = X_test.reshape(-1,2,128,1)
#==============================================================================
# Hyper parameters initializaion
num_epochs = 100
minibatch_size = 1024
num_train_ex = 110000
num_batches = 108 #int(num_train_ex / minibatch_size)
print(num_batches)
paddings = tf.constant([[0,0],[0,0],[2,2],[0,0]])



#==============================================================================
# create the model
def conv_net(x_in,is_training):
    #x = tf.reshape(x_in,[None,1,2,128], name='reshaped_input') # BCHW
    print(x_in)
    pad1 = tf.pad(x_in, paddings, mode='CONSTANT', name='padding1', constant_values=0.)
    print('this is pad1 shape', pad1.shape)
    
    conv1 = tf.layers.conv2d(pad1, 256, [1,3], strides=(1, 1), padding='valid', data_format='channels_last',
                             activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.glorot_uniform_initializer(seed=None, dtype=tf.float32),
                             trainable=True, name='conv1')
    print('this is conv1 shape', conv1.shape)
    
    pad2 = tf.pad(conv1, paddings, mode='CONSTANT', name='padding2', constant_values=0)
    print('this is pad2 shape', pad2.shape)
    
    
    conv2 = tf.layers.conv2d(pad2, 80, [2,3], strides=(1, 1), padding='valid', data_format='channels_last',
                             activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.glorot_uniform_initializer(seed=None, dtype=tf.float32),
                             trainable=True, name='conv2')
    print('this is conv2 shape', conv2.shape)
    
    dropou1 = tf.layers.dropout(conv2, rate=0.5, noise_shape=None, training=is_training, name='dropou1')
    
    flattened = tf.contrib.layers.flatten(dropou1)
    print('this is flattened shape', flattened.shape)
    
    dense1 = tf.layers.dense(flattened, 256, activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.keras.initializers.he_normal(seed=None), name='dense1')
    print('this is dense1 shape', dense1.shape)
    
    dropou2 = tf.layers.dropout(dense1, rate=0.5, noise_shape=None, training=is_training, name='dropou2')
    
    dense2_logits = tf.layers.dense(dropou2, 11, use_bias=True,
                                    kernel_initializer=tf.keras.initializers.he_normal(seed=None), name='dense2_logits')
    print('this is dense2 shape', dense2_logits.shape)
    
    return dense2_logits
#==============================================================================
X = tf.placeholder(tf.float32,[None,2,128,1], name='X')
Y = tf.placeholder(tf.float32,[None,11], name='Y')
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

#==============================================================================
# construct the logits, cots, optimizer, accuracy, and initialization
logits_withoutname = conv_net(X,is_training)
logits = tf.identity(logits_withoutname,name='logits')


prediction = tf.nn.softmax(logits, name = 'predictions')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y), name='cost')

optimizer = tf.train.AdamOptimizer().minimize(cost) 

accuracy = tf.reduce_mean( tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1)),tf.float32), name = 'accuracy' )


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

init = tf.global_variables_initializer()



# Start training# Start 
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for epoch_counter in range(num_epochs):
        print(epoch_counter)
        start = 0
        for batch_counter in range(num_batches):
            if batch_counter< 107:
                x_batch = X_train_reshaped[start:start+minibatch_size] # Remember to put each minibatch within the minibatch counter
                y_batch = Y_train[start:start+minibatch_size]
                start = start+minibatch_size
            else:
                x_batch = X_train_reshaped[start:110000] # Remember to put each minibatch within the minibatch counter
                y_batch = Y_train[start:110000]
            #print(batch_counter)
            sess.run(optimizer,feed_dict={X: x_batch, Y: y_batch, is_training: True})
        #if epoch_counter % 1 == 0:
        sum_test_cost = 0
        sum_test_accuracy = 0
        start = 0
        for batch_counter in range(num_batches):
            if batch_counter< 107:
                x_batch_test = X_test_reshaped[start:start+minibatch_size] # Remember to put each minibatch within the minibatch counter
                y_batch_test = Y_test[start:start+minibatch_size]
                start = start+minibatch_size
            else:
                x_batch_test = X_test_reshaped[start:110000] # Remember to put each minibatch within the minibatch counter
                y_batch_test = Y_test[start:110000]
            
            adj_coef = len(y_batch_test)/minibatch_size
            test_cost, test_accuracy = sess.run([cost, accuracy],feed_dict={X:x_batch_test,Y:y_batch_test, is_training: False})
            sum_test_cost = sum_test_cost + adj_coef * test_cost
            sum_test_accuracy = sum_test_accuracy + adj_coef * test_accuracy
            
        print("Test cost:", sum_test_cost/num_batches,"    Test accuracy:", sum_test_accuracy/num_batches)





    # Save the variables(model weights) to disk.
    # Remember that Tensorflow variables are only alive inside a session.
    # So, you have to save the model inside a session by calling save method on saver object we created.
    # Here, sess is the session object, while ‘my-test-model’ is the name you want to give your model
#==============================================================================
    save_path = saver.save(sess, "ModulationCls_GPU/tfTrainedmodel")
    print("Model saved in path: %s" % save_path)

































































