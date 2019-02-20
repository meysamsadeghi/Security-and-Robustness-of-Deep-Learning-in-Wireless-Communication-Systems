# all aoutoencoder classes are here
import numpy as np
import tensorflow as tf


###############################  MLP of Table 1 ###############################
class AE_MLP(object):
    def __init__(self, k, n, seed=None, filename=None):
        self.k = k 
        self.n = n
        self.bits_per_symbol = self.k/self.n
        self.M = 2**self.k
        self.seed = seed            
        self.graph = None  
        self.sess = None  
        self.vars = None  
        self.saver = None 
        self.constellations = None
        self.blers = None
        self.create_graph() 
        self.create_session()
        if filename is not None:    
            self.load(filename)       
        return
    
    def create_graph(self):
        '''This function creates the computation graph of the autoencoder'''
        self.graph = tf.Graph()        
        with self.graph.as_default():  
            tf.set_random_seed(self.seed) 
            batch_size = tf.placeholder(tf.int32, shape=())
            
            # Transmitter
            s = tf.random_uniform(shape=[batch_size], minval=0, maxval=self.M, dtype=tf.int64)
            x = self.encoder(s)     
            
            # the attack vector
            p = tf.placeholder(tf.float32,shape=(None,2,self.n)) 
            
            # Channel
            noise_std = tf.placeholder(tf.float32, shape=()) # 
            noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=noise_std, seed=self.seed)
            y = x + noise + p
    
            # Receiver
            s_hat = self.decoder(y)
            
            # Loss function
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=s, logits=s_hat)
            
            # Performance metrics
            correct_predictions = tf.equal(tf.argmax(tf.nn.softmax(s_hat), axis=1), s)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            bler = 1-accuracy
            
            # Optimizer
            lr = tf.placeholder(tf.float32, shape=())    
            train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        
            # References to graph variables we need to access later 
            self.vars = {
                'accuracy': accuracy,
                'batch_size': batch_size,
                'bler': bler,
                'cross_entropy': cross_entropy,
                'init': tf.global_variables_initializer(),
                'lr': lr,
                'noise_std': noise_std,
                'noise': noise,
                'p': p,
                's': s,
                's_hat': s_hat,
                'train_op': train_op,
                'x': x,
                'y': y,
            }            
            self.saver = tf.train.Saver()
        return
    
    def create_session(self):
        '''Create a session for the autoencoder instance with the compuational graph'''
        self.sess = tf.Session(graph=self.graph)        
        self.sess.run(self.vars['init'])
        return
    
    def encoder(self, input):
        '''The transmitter'''
        W = self.weight_variable((self.M,self.M))
        x = tf.nn.elu(tf.nn.embedding_lookup(W, input))
        x = tf.layers.dense(x, 2*self.n, activation=None)
        x = tf.reshape(x, shape=[-1,2,self.n])
        #Average power normalization
        x = x/tf.sqrt(2*tf.reduce_mean(tf.square(x)))
        return x
    
    def decoder(self, input):
        '''The Receiver'''
        y = tf.reshape(input, shape=[-1,2*self.n])
        y = tf.layers.dense(y, self.M, activation=tf.nn.relu)
        y = tf.layers.dense(y, self.M, activation=None)
        return y

    def EbNo2Sigma(self, ebnodb):
        '''Convert Eb/No in dB to noise standard deviation'''
        ebno = 10**(ebnodb/10)
        return 1/np.sqrt(2*self.bits_per_symbol*ebno) 
    
    def gen_feed_dict(self, perturbation, batch_size, ebnodb, lr):
        '''Generate a feed dictionary for training and validation'''        
        return {
            self.vars['p']: perturbation,
            self.vars['batch_size']: batch_size,
            self.vars['noise_std']: self.EbNo2Sigma(ebnodb),
            self.vars['lr']: lr,
        }           

    def load(self, filename):
        '''Load a pre_trained model'''
        return self.saver.restore(self.sess, filename)
    
    def save(self, filename):
        '''Save the current model'''
        return self.saver.save(self.sess, filename)  
    
    def test_step(self, p, batch_size, ebnodb):
        '''Compute the BLER over a single batch and Eb/No'''
        bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr=0))
        return bler
    
    def transmit(self, s):
        '''Returns the transmitted sigals corresponding to message indices'''
        return self.sess.run(self.vars['x'], feed_dict={self.vars['s']: s})
       
    def train(self, p, training_params, validation_params):  
        '''Training and validation loop'''
        for index, params in enumerate(training_params):            
            batch_size, lr, ebnodb, iterations = params            
            print('\nBatch Size: ' + str(batch_size) +
                  ', Learning Rate: ' + str(lr) +
                  ', EbNodB: ' + str(ebnodb) +
                  ', Iterations: ' + str(iterations))
            
            val_size, val_ebnodb, val_steps = validation_params[index]
            for i in range(iterations):
                self.train_step(p, batch_size, ebnodb, lr)    
                if (i%val_steps==0):
                    bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(p,val_size, val_ebnodb, lr))
                    print(bler)                           
        return       
    
    def train_step(self, p, batch_size, ebnodb, lr):
        '''A single training step'''
        self.sess.run(self.vars['train_op'], feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr)) #self.sess.run(train_op, feed_dict=self.gen_feed_dict(batch_size, ebnodb, lr))#s
        return 
    
    def weight_variable(self, shape):
        '''Xavier-initialized weights optimized for ReLU Activations'''
        (fan_in, fan_out) = shape
        low = np.sqrt(6.0/(fan_in + fan_out)) 
        high = -np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    

    def bler_sim_attack_AWGN(self, p, PSR_dB, ebnodbs, batch_size, iterations):
        '''Generate the BLER for 3 cases: 1) no attack, 2) adversarial attack, and 3) jamming attack'''
        np.random.seed(seed=self.seed)
        PSR = 10**(PSR_dB/10)
        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p)**2 +  0.00000001) ) # 
        p = scale_factor * p
        BLER_no_attack = np.zeros_like(ebnodbs)
        BLER_adv_attack = np.zeros_like(ebnodbs)
        BLER_jamming = np.zeros_like(ebnodbs)
        for i in range(iterations):
            # No attack - clean case
            bler = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(np.zeros([1,2,self.n]), batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) #bler = np.array([self.sess.run(self.vars['bler'],feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs])
            BLER_no_attack = BLER_no_attack + bler/iterations
            # adversarial attack
            bler_attack = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(  p ,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_adv_attack = BLER_adv_attack + bler_attack/iterations
            # Jamming attack
            normal_noise_as_jammer = np.random.normal(0,1,p.shape)
            jamming = np.linalg.norm(p) * (1 / np.linalg.norm(normal_noise_as_jammer)) * normal_noise_as_jammer
            bler_jamming= np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(jamming,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_jamming = BLER_jamming + bler_jamming/iterations
        return BLER_no_attack, BLER_adv_attack, BLER_jamming


    def fgm_attack(self,s,p, ebnodb): #in_img,in_label,num_class
        '''Create an input specific adversarial example using the method proposed by Sadeghi and Larsson in [2] '''
        np.random.seed(seed=self.seed)
        num_class = self.M 
        y_reshaped = np.reshape(self.sess.run(self.vars['y'], feed_dict={self.vars['s']: s, self.vars['p']: p, self.vars['noise_std']: self.EbNo2Sigma(ebnodb)}), [1,2,self.n])  #[-1]         #print('y_reshaped',y_reshaped.shape)        
        eps_acc = 0.0000001 * np.linalg.norm(y_reshaped)
        epsilon_vector = np.zeros([num_class])
        predictions = tf.nn.softmax(self.vars['s_hat'], name = 'predictions')        
        for cls in range(num_class):
            s_target = np.array([cls])
            adv_per_needtoreshape  = -1 * np.asarray( self.sess.run(tf.gradients(self.vars['cross_entropy'],self.vars['y']), feed_dict={self.vars['y']: y_reshaped , self.vars['s']:s_target }) )
            adv_per = adv_per_needtoreshape.reshape(1,2,self.n)
            norm_adv_per = adv_per / (np.linalg.norm(adv_per) +  0.000000000001)
            epsilon_max = 1 * np.linalg.norm(y_reshaped)
            epsilon_min = 0
            num_iter = 0
            wcount = 0
            while (epsilon_max-epsilon_min > eps_acc) and (num_iter < 30):
                wcount = wcount+1
                num_iter = num_iter +1
                epsilon = (epsilon_max + epsilon_min)/2
                adv_img_givencls = y_reshaped + (epsilon * norm_adv_per)
                
                predicted_probabilities = self.sess.run(predictions, feed_dict={self.vars['y']: adv_img_givencls})
                compare = np.equal(np.argmax(predicted_probabilities),s)
                if compare:
                    epsilon_min = epsilon
                else:
                    epsilon_max = epsilon
            epsilon_vector[cls] = epsilon + eps_acc
        false_cls = np.argmin(epsilon_vector)
        minimum_epsilon = np.min(epsilon_vector)
        adv_dirc = -1 * np.asarray(self.sess.run(tf.gradients(self.vars['cross_entropy'],self.vars['y']), feed_dict={self.vars['y']: y_reshaped, self.vars['s']: np.asarray([false_cls]) })  ).reshape(1,2,self.n)
        norm_adv_dirc = adv_dirc / (np.linalg.norm(adv_dirc) + 0.000000000001)
        adv_perturbation = minimum_epsilon * norm_adv_dirc
        return adv_perturbation, false_cls, minimum_epsilon

    
        
    def UAPattack_fgm(self,ebnodb,num_samples,PSR_dB):
        '''Create a Universal Adversarial Perturbation as suggested by Alg. 1 of Sadeghi et al in [2]'''
        np.random.seed(seed=self.seed)
        universal_per_fgm = np.zeros([1,2,self.n])
        for cnr_index in range(num_samples):#               
            s =  np.asarray([np.floor(np.random.uniform(0,16,1))]).reshape(1) 
            predicted_label = np.argmax( self.sess.run(self.vars['s_hat'], feed_dict={self.vars['s']:s, self.vars['p']:universal_per_fgm, self.vars['noise_std']: self.EbNo2Sigma(ebnodb)}) )
            if predicted_label == s:
                # First we need to find adverssarial direction for this instant  by solving eq. (1) of the paper
                adv_perturbation,_,_ = self.fgm_attack(s, universal_per_fgm,ebnodb)
                adv_perturbn_reshaped = adv_perturbation.reshape([1,2,self.n])
                UAP = universal_per_fgm + adv_perturbation.reshape([1,2,self.n])
                PSR = 10**(PSR_dB/10)
                Epsilon_uni = np.sqrt( (PSR * self.n) / (np.linalg.norm(UAP)**2 + 0.00000001) )
                # Second we need to revise the universal perturbation
                if np.linalg.norm(universal_per_fgm + adv_perturbn_reshaped) < Epsilon_uni: 
                    universal_per_fgm = universal_per_fgm + adv_perturbn_reshaped
                else:
                    universal_per_fgm =  Epsilon_uni * (universal_per_fgm + adv_perturbn_reshaped) 
        return universal_per_fgm


###############################  CNN of Table 1 ###############################
class AE_CNN(object):
    def __init__(self, k, n, seed=None, filename=None):
        self.k = k 
        self.n = n
        self.bits_per_symbol = self.k/self.n
        self.M = 2**self.k
        self.seed = seed
        self.graph = None  
        self.sess = None  
        self.vars = None 
        self.saver = None 
        self.constellations = None
        self.blers = None
        self.create_graph()
        self.create_session()
        if filename is not None:    
            self.load(filename)       
        return
    
    def create_graph(self):
        '''This function creates the computation graph of the autoencoder'''
        self.graph = tf.Graph()        
        with self.graph.as_default():  
            tf.set_random_seed(self.seed)
            batch_size = tf.placeholder(tf.int32, shape=(), name='batchsize')
            is_training = tf.placeholder(tf.bool, name='istraining')
            dr_out = tf.placeholder(tf.float32,shape=(), name='drout')
            # Transmitter
            s = tf.random_uniform(shape=[batch_size], minval=0, maxval=self.M, dtype=tf.int64)
            x = self.encoder(s)     

            # the attack vector
            p = tf.placeholder(tf.float32,shape=(None,2,self.n), name='pname') # batch * 2 * n is the shape of y and x.
            
            # Channel
            noise_std = tf.placeholder(tf.float32, shape=()) 
            noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=noise_std)
            y = x + noise + p
            
            # Receiver
            s_hat = self.decoder(y, dr_out, is_training)
            
            # Loss function
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=s, logits=s_hat)
 
            # Performance metrics
            correct_predictions = tf.equal(tf.argmax(tf.nn.softmax(s_hat), axis=1), s)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            bler = 1-accuracy
            
            
            # Optimizer
            lr = tf.placeholder(tf.float32, shape=())      
            train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        
            # References to graph variables we need to access later 
            self.vars = {
                'accuracy': accuracy,
                'batch_size': batch_size,
                'bler': bler,
                'cross_entropy': cross_entropy,
                'dr_out':dr_out,
                'init': tf.global_variables_initializer(),
                'is_training':is_training,
                'lr': lr,
                'noise_std': noise_std,
                'noise': noise,
                'p': p,
                's': s,
                's_hat': s_hat,
                'train_op': train_op,
                'x': x,
                'y': y,
            }            
            self.saver = tf.train.Saver()
        return
    
    
    def create_session(self):
        '''Create a session for the autoencoder instance with the compuational graph'''
        self.sess = tf.Session(graph=self.graph)      
        self.sess.run(self.vars['init'])
        return
    
    
    def encoder(self, input):
        '''The transmitter'''
        W = self.weight_variable((self.M,self.M))
        x = tf.nn.elu(tf.nn.embedding_lookup(W, input)) 
        x = tf.reshape(x,[-1,1,self.M])
        conv0 = tf.layers.conv1d(x, 16, 6, strides=1, padding='same', data_format='channels_first',
                             activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.glorot_uniform_initializer(seed=None, dtype=tf.float32),
                             trainable=True)
        flattened0 = tf.contrib.layers.flatten(conv0)
        x = tf.layers.dense(flattened0, 2*self.n, activation=None)
        x = tf.reshape(x, shape=[-1,2,self.n]) 
        #Average power normalization
        x = x/tf.sqrt(2*tf.reduce_mean(tf.square(x)))
        return x
    
    def decoder(self, input, dr_out, is_training):
        '''The Receiver'''
        reshaped = tf.reshape(input, shape=[-1,1,2,self.n])
        conv1 = tf.layers.conv2d(reshaped, 16, [2,3], strides=(1, 1), padding='same', data_format='channels_first',
                             activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.glorot_uniform_initializer(seed=None, dtype=tf.float32),
                             trainable=True)
        conv2 = tf.layers.conv2d(conv1, 8, [2,3], strides=(1, 1), padding='same', data_format='channels_first',
                             activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.glorot_uniform_initializer(seed=None, dtype=tf.float32),
                             trainable=True)
        drout = tf.layers.dropout(conv2, rate=dr_out, noise_shape=None, training=is_training, name='dropou1')
        flattened = tf.contrib.layers.flatten(drout)
        dense1 = tf.layers.dense(flattened, 2*self.M, activation=tf.nn.relu)
        y = tf.layers.dense(dense1, self.M, activation=None)
        return y

    def EbNo2Sigma(self, ebnodb):
        '''Convert Eb/No in dB to noise standard deviation'''
        ebno = 10**(ebnodb/10)
        return 1/np.sqrt(2*self.bits_per_symbol*ebno) 
    
    def gen_feed_dict(self, is_training,dr_out, perturbation, batch_size, ebnodb, lr):
        '''Generate a feed dictionary for training and validation'''      
        return {
            self.vars['is_training']: is_training,
            self.vars['dr_out']: dr_out,
            self.vars['p']: perturbation,
            self.vars['batch_size']: batch_size,
            self.vars['noise_std']: self.EbNo2Sigma(ebnodb),
            self.vars['lr']: lr,
        }    
    
    
    def load(self, filename):
        '''Load a pre_trained model'''
        return self.saver.restore(self.sess, filename)
    
    def save(self, filename):
        '''Save the current model'''
        return self.saver.save(self.sess, filename)  
    
    def test_step(self, is_training, dr_out, p, batch_size, ebnodb):
        '''Compute the BLER over a single batch and Eb/No'''
        bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(is_training, dr_out, p, batch_size, ebnodb, lr=0))  
        return bler
    
    def transmit(self, s):
        '''Returns the transmitted sigals corresponding to message indices'''
        return self.sess.run(self.vars['x'], feed_dict={self.vars['s']: s})
       
    def train(self, is_training, dr_out ,p, training_params, validation_params):  
        '''Training and validation loop'''
        for index, params in enumerate(training_params):            
            batch_size, lr, ebnodb, iterations = params            
            print('\nBatch Size: ' + str(batch_size) +
                  ', Learning Rate: ' + str(lr) +
                  ', EbNodB: ' + str(ebnodb) +
                  ', Iterations: ' + str(iterations))
            
            val_size, val_ebnodb, val_steps = validation_params[index]
            
            for i in range(iterations):
                self.train_step(is_training, dr_out, p, batch_size, ebnodb, lr)    
                if (i%val_steps==0):
                    bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(is_training, dr_out , p,val_size, val_ebnodb, lr))
                    print(bler)                           
        return       
    
    def train_step(self, is_training, dr_out , p, batch_size, ebnodb, lr):
        '''A single training step'''
        self.sess.run(self.vars['train_op'], feed_dict=self.gen_feed_dict(is_training, dr_out , p, batch_size, ebnodb, lr)) #self.sess.run(train_op, feed_dict=self.gen_feed_dict(batch_size, ebnodb, lr))#s
        return 
    
    def weight_variable(self, shape):
        '''Xavier-initialized weights optimized for ReLU Activations'''
        (fan_in, fan_out) = shape
        low = np.sqrt(6.0/(fan_in + fan_out)) 
        high = -np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    
    
    def bler_sim_attack_AWGN(self, is_training, dr_out , p, PSR_dB, ebnodbs, batch_size, iterations):
        '''Generate the BLER for 4 cases: 1) no attack, 2) synchronous adversarial attack, 3) non-synchronous adversarial attack and 4) jamming attack'''
        PSR = 10**(PSR_dB/10)
        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p)**2 + 0.00000001) ) # note that self.n is the power of the x, as designed by Jakob
        p = scale_factor * p
        
        BLER_no_attack = np.zeros_like(ebnodbs)
        BLER_attack_rolled = np.zeros_like(ebnodbs)
        BLER_jamming = np.zeros_like(ebnodbs)

        for i in range(iterations):
            # No attack - clean case
            bler = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out , np.zeros([1,2,self.n]), batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) #bler = np.array([self.sess.run(self.vars['bler'],feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs])
            BLER_no_attack = BLER_no_attack + bler/iterations
            # attack - rolled attack - nonsynchronous
            p_rolled = np.roll(p, int(np.ceil(np.random.uniform(0,self.n))))
            bler_attack_rolled = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out ,p_rolled,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_attack_rolled = BLER_attack_rolled + bler_attack_rolled/iterations
            # Jamming attack
            normal_noise_as_jammer = np.random.normal(0,1,p.shape)
            jamming = np.linalg.norm(p) * (1 / np.linalg.norm(normal_noise_as_jammer)) * normal_noise_as_jammer
            bler_jamming= np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out, jamming,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_jamming = BLER_jamming + bler_jamming/iterations
        return BLER_no_attack, BLER_attack_rolled, BLER_jamming
    
    
    
    
    
    
    
    
    
    
################################################################################
################################################################################
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
############################################################################### 
    
#################### MLP of Table 2 == netOne_MLP #############################
class AE_netOne_MLP(object):
    def __init__(self, k, n, seed=None, filename=None):
        self.k = k 
        self.n = n
        self.bits_per_symbol = self.k/self.n
        self.M = 2**self.k
        self.seed = seed         
        self.graph = None  
        self.sess = None  
        self.vars = None  
        self.saver = None 
        self.constellations = None
        self.blers = None
        self.create_graph() 
        self.create_session()
        if filename is not None:    
            self.load(filename)       
        return
    
    def create_graph(self):
        '''This function creates the computation graph of the autoencoder'''
        self.graph = tf.Graph()        
        with self.graph.as_default():  
            tf.set_random_seed(self.seed) 
            batch_size = tf.placeholder(tf.int32, shape=())
            
            # Transmitter
            s = tf.random_uniform(shape=[batch_size], minval=0, maxval=self.M, dtype=tf.int64)
            x = self.encoder(s)     
            
            # the attack vector
            p = tf.placeholder(tf.float32,shape=(None,2,self.n)) 
            
            # Channel
            noise_std = tf.placeholder(tf.float32, shape=()) # 
            noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=noise_std, seed=self.seed)
            y = x + noise + p
    
            # Receiver
            s_hat = self.decoder(y)
            
            # Loss function
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=s, logits=s_hat)
            
            # Performance metrics
            correct_predictions = tf.equal(tf.argmax(tf.nn.softmax(s_hat), axis=1), s)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            bler = 1-accuracy
            
            # Optimizer
            lr = tf.placeholder(tf.float32, shape=())    
            train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        
            # References to graph variables we need to access later 
            self.vars = {
                'accuracy': accuracy,
                'batch_size': batch_size,
                'bler': bler,
                'cross_entropy': cross_entropy,
                'init': tf.global_variables_initializer(),
                'lr': lr,
                'noise_std': noise_std,
                'noise': noise,
                'p': p,
                's': s,
                's_hat': s_hat,
                'train_op': train_op,
                'x': x,
                'y': y,
            }            
            self.saver = tf.train.Saver()
        return
    
    def create_session(self):
        '''Create a session for the autoencoder instance with the compuational graph'''
        self.sess = tf.Session(graph=self.graph)        
        self.sess.run(self.vars['init'])
        return
    
    def encoder(self, input):
        '''The transmitter'''
        W = self.weight_variable((self.M,self.M))
        x = tf.nn.relu(tf.nn.embedding_lookup(W, input))
        x = tf.layers.dense(x, 2*self.n, activation=None)
        x = tf.reshape(x, shape=[-1,2,self.n])
        #Average power normalization
        x = x/tf.sqrt(2*tf.reduce_mean(tf.square(x)))
        return x
    
    def decoder(self, input):
        '''The Receiver'''
        y = tf.reshape(input, shape=[-1,2*self.n])
        y = tf.layers.dense(y, self.M, activation=tf.nn.relu)
        y = tf.layers.dense(y, self.M, activation=None)
        return y

    def EbNo2Sigma(self, ebnodb):
        '''Convert Eb/No in dB to noise standard deviation'''
        ebno = 10**(ebnodb/10)
        return 1/np.sqrt(2*self.bits_per_symbol*ebno) 
    
    def gen_feed_dict(self, perturbation, batch_size, ebnodb, lr):
        '''Generate a feed dictionary for training and validation'''        
        return {
            self.vars['p']: perturbation,
            self.vars['batch_size']: batch_size,
            self.vars['noise_std']: self.EbNo2Sigma(ebnodb),
            self.vars['lr']: lr,
        }           

    def load(self, filename):
        '''Load a pre_trained model'''
        return self.saver.restore(self.sess, filename)
    
    def save(self, filename):
        '''Save the current model'''
        return self.saver.save(self.sess, filename)  
    
    def test_step(self, p, batch_size, ebnodb):
        '''Compute the BLER over a single batch and Eb/No'''
        bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr=0))
        return bler
    
    def transmit(self, s):
        '''Returns the transmitted sigals corresponding to message indices'''
        return self.sess.run(self.vars['x'], feed_dict={self.vars['s']: s})
       
    def train(self, p, training_params, validation_params):  
        '''Training and validation loop'''
        for index, params in enumerate(training_params):            
            batch_size, lr, ebnodb, iterations = params            
            print('\nBatch Size: ' + str(batch_size) +
                  ', Learning Rate: ' + str(lr) +
                  ', EbNodB: ' + str(ebnodb) +
                  ', Iterations: ' + str(iterations))
            
            val_size, val_ebnodb, val_steps = validation_params[index]
            for i in range(iterations):
                self.train_step(p, batch_size, ebnodb, lr)    
                if (i%val_steps==0):
                    bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(p,val_size, val_ebnodb, lr))
                    print(bler)                           
        return       
    
    def train_step(self, p, batch_size, ebnodb, lr):
        '''A single training step'''
        self.sess.run(self.vars['train_op'], feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr)) #self.sess.run(train_op, feed_dict=self.gen_feed_dict(batch_size, ebnodb, lr))#s
        return 
    
    def weight_variable(self, shape):
        '''Xavier-initialized weights optimized for ReLU Activations'''
        (fan_in, fan_out) = shape
        low = np.sqrt(6.0/(fan_in + fan_out)) 
        high = -np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    

    def bler_sim_attack_AWGN(self, p, PSR_dB, ebnodbs, batch_size, iterations):
        '''Generate the BLER for 3 cases: 1) no attack, 2) adversarial attack, and 3) jamming attack'''
        np.random.seed(seed=self.seed)
        PSR = 10**(PSR_dB/10)
        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p)**2 +  0.00000001) ) # 
        p = scale_factor * p
        BLER_no_attack = np.zeros_like(ebnodbs)
        BLER_attack_rolled = np.zeros_like(ebnodbs)
        BLER_jamming = np.zeros_like(ebnodbs)
        for i in range(iterations):
            # No attack - clean case
            bler = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(np.zeros([1,2,self.n]), batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) #bler = np.array([self.sess.run(self.vars['bler'],feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs])
            BLER_no_attack = BLER_no_attack + bler/iterations
            # attack - rolled attack - nonsynchronous
            p_rolled = np.roll(p, int(np.ceil(np.random.uniform(0,self.n))))
            bler_attack_rolled = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(p_rolled,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_attack_rolled = BLER_attack_rolled + bler_attack_rolled/iterations
            # Jamming attack
            normal_noise_as_jammer = np.random.normal(0,1,p.shape)
            jamming = np.linalg.norm(p) * (1 / np.linalg.norm(normal_noise_as_jammer)) * normal_noise_as_jammer
            bler_jamming= np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(jamming,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_jamming = BLER_jamming + bler_jamming/iterations
        return BLER_no_attack, BLER_attack_rolled, BLER_jamming


#################### Deeper MLP of Table 2 == netTwo_DeepMLP #############################
class AE_netTwo_DeepMLP(object):
    def __init__(self, k, n, seed=None, filename=None):
        self.k = k 
        self.n = n
        self.bits_per_symbol = self.k/self.n
        self.M = 2**self.k
        self.seed = seed          
        self.graph = None  
        self.sess = None  
        self.vars = None  
        self.saver = None 
        self.constellations = None
        self.blers = None
        self.create_graph() 
        self.create_session()
        if filename is not None:    
            self.load(filename)       
        return
    
    def create_graph(self):
        '''This function creates the computation graph of the autoencoder'''
        self.graph = tf.Graph()        
        with self.graph.as_default():  
            tf.set_random_seed(self.seed) 
            batch_size = tf.placeholder(tf.int32, shape=())
            is_training = tf.placeholder(tf.bool, name='istraining')
            dr_out = tf.placeholder(tf.float32,shape=(), name='drout')
            # Transmitter
            s = tf.random_uniform(shape=[batch_size], minval=0, maxval=self.M, dtype=tf.int64)
            x = self.encoder(s,dr_out, is_training)     
            
            # the attack vector
            p = tf.placeholder(tf.float32,shape=(None,2,self.n)) 
            
            # Channel
            noise_std = tf.placeholder(tf.float32, shape=()) # 
            noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=noise_std, seed=self.seed)
            y = x + noise + p
    
            # Receiver
            s_hat = self.decoder(y,dr_out, is_training)
            
            # Loss function
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=s, logits=s_hat)
            
            # Performance metrics
            correct_predictions = tf.equal(tf.argmax(tf.nn.softmax(s_hat), axis=1), s)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            bler = 1-accuracy
            
            # Optimizer
            lr = tf.placeholder(tf.float32, shape=())    
            train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        
            # References to graph variables we need to access later 
            self.vars = {
                'accuracy': accuracy,
                'batch_size': batch_size,
                'bler': bler,
                'cross_entropy': cross_entropy,
                'dr_out':dr_out,
                'init': tf.global_variables_initializer(),
                'is_training':is_training,
                'lr': lr,
                'noise_std': noise_std,
                'noise': noise,
                'p': p,
                's': s,
                's_hat': s_hat,
                'train_op': train_op,
                'x': x,
                'y': y,
            }            
            self.saver = tf.train.Saver()
        return
    
    def create_session(self):
        '''Create a session for the autoencoder instance with the compuational graph'''
        self.sess = tf.Session(graph=self.graph)       
        self.sess.run(self.vars['init'])
        return
    
    def encoder(self, input, dr_out, is_training):
        '''The transmitter'''
        W = self.weight_variable((self.M,self.M))
        x = tf.nn.relu(tf.nn.embedding_lookup(W, input))
        x = tf.layers.dense(x, 2*self.n, activation=tf.nn.relu)
        x = tf.layers.dropout(x,rate=dr_out, training = is_training)
        x = tf.layers.dense(x, 4*self.n, activation=tf.nn.relu)
        x = tf.layers.dropout(x,rate=dr_out, training = is_training)
#        x = tf.layers.dense(x, 4*self.n, activation=tf.nn.relu)
#        x = tf.layers.dropout(x,rate=dr_out, training = is_training)
        x = tf.layers.dense(x, 2*self.n, activation=None)
        x = tf.reshape(x, shape=[-1,2,self.n])
        #Average power normalization
        x = x/tf.sqrt(2*tf.reduce_mean(tf.square(x)))
        return x
    
    def decoder(self, input,  dr_out, is_training):
        '''The Receiver'''
        y = tf.reshape(input, shape=[-1,2*self.n])
        y = tf.layers.dense(y, 2*self.M, activation=tf.nn.relu)
        y = tf.layers.dropout(y,rate=dr_out, training=is_training)
        y = tf.layers.dense(y, 4*self.M, activation=tf.nn.relu)
        y = tf.layers.dropout(y,rate=dr_out, training=is_training)
#        y = tf.layers.dense(y, 4*self.M, activation=tf.nn.relu)
#        y = tf.layers.dropout(y,rate=dr_out, training=is_training)
        y = tf.layers.dense(y, 2*self.M, activation=tf.nn.relu)
        y = tf.layers.dropout(y,rate=dr_out, training=is_training)
        y = tf.layers.dense(y, self.M, activation=None)
        return y

    def EbNo2Sigma(self, ebnodb):
        '''Convert Eb/No in dB to noise standard deviation'''
        ebno = 10**(ebnodb/10)
        return 1/np.sqrt(2*self.bits_per_symbol*ebno) 
    
    def gen_feed_dict(self, is_training,dr_out, perturbation, batch_size, ebnodb, lr):
        '''Generate a feed dictionary for training and validation'''        
        return {
            self.vars['is_training']: is_training,
            self.vars['dr_out']: dr_out,
            self.vars['p']: perturbation,
            self.vars['batch_size']: batch_size,
            self.vars['noise_std']: self.EbNo2Sigma(ebnodb),
            self.vars['lr']: lr,
        }    
    
    def load(self, filename):
        '''Load a pre_trained model'''
        return self.saver.restore(self.sess, filename)
    
    def save(self, filename):
        '''Save the current model'''
        return self.saver.save(self.sess, filename)  
    
    def test_step(self, is_training, dr_out, p, batch_size, ebnodb):
        '''Compute the BLER over a single batch and Eb/No'''
        bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(is_training, dr_out, p, batch_size, ebnodb, lr=0))
        return bler
    
    def transmit(self, s):
        '''Returns the transmitted sigals corresponding to message indices'''
        return self.sess.run(self.vars['x'], feed_dict={self.vars['s']: s})
       
    def train(self, is_training, dr_out , p, training_params, validation_params):  
        '''Training and validation loop'''
        for index, params in enumerate(training_params):            
            batch_size, lr, ebnodb, iterations = params            
            print('\nBatch Size: ' + str(batch_size) +
                  ', Learning Rate: ' + str(lr) +
                  ', EbNodB: ' + str(ebnodb) +
                  ', Iterations: ' + str(iterations))
            
            val_size, val_ebnodb, val_steps = validation_params[index]
            for i in range(iterations):
                self.train_step(is_training, dr_out, p, batch_size, ebnodb, lr)    
                if (i%val_steps==0):
                    bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(is_training, dr_out, p,val_size, val_ebnodb, lr))
                    print(bler)                           
        return       
    
    def train_step(self, is_training, dr_out , p, batch_size, ebnodb, lr):
        '''A single training step'''
        self.sess.run(self.vars['train_op'], feed_dict=self.gen_feed_dict(is_training, dr_out , p, batch_size, ebnodb, lr)) 
        return 
    
    def weight_variable(self, shape):
        '''Xavier-initialized weights optimized for ReLU Activations'''
        (fan_in, fan_out) = shape
        low = np.sqrt(6.0/(fan_in + fan_out)) 
        high = -np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    

    def bler_sim_attack_AWGN(self, is_training, dr_out , p, PSR_dB, ebnodbs, batch_size, iterations):
        np.random.seed(seed=self.seed)
        PSR = 10**(PSR_dB/10)
        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p)**2 +  0.00000001) ) 
        p = scale_factor * p
        BLER_no_attack = np.zeros_like(ebnodbs)
        BLER_attack_rolled = np.zeros_like(ebnodbs)
        BLER_jamming = np.zeros_like(ebnodbs)
        for i in range(iterations):
            # No attack - clean case
            bler = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out , np.zeros([1,2,self.n]), batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) 
            BLER_no_attack = BLER_no_attack + bler/iterations
            # attack - rolled attack - nonsynchronous
            p_rolled = np.roll(p, int(np.ceil(np.random.uniform(0,self.n))))
            bler_attack_rolled = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out , p_rolled,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_attack_rolled = BLER_attack_rolled + bler_attack_rolled/iterations
            # Jamming attack
            normal_noise_as_jammer = np.random.normal(0,1,p.shape)
            jamming = np.linalg.norm(p) * (1 / np.linalg.norm(normal_noise_as_jammer)) * normal_noise_as_jammer
            bler_jamming= np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out , jamming,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) 
            BLER_jamming = BLER_jamming + bler_jamming/iterations
        return BLER_no_attack, BLER_attack_rolled, BLER_jamming

################################################################################
################################################################################
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
############################################################################### 
class AE_inf_rate(object):
    def __init__(self, k, n, seed=None, filename=None):
        self.k = k 
        self.n = n
        self.bits_per_symbol = self.k/self.n
        self.M = 2**self.k
        self.seed = seed      
        self.graph = None # 
        self.sess = None  # 
        self.vars = None  # 
        self.saver = None # 
        self.constellations = None
        self.blers = None
        self.create_graph() #
        self.create_session()
        if filename is not None:    
            self.load(filename)       
        return
    
    def create_graph(self):
        '''This function creates the computation graph of the autoencoder'''
        self.graph = tf.Graph()        
        with self.graph.as_default():  #
            tf.set_random_seed(self.seed) # 
            batch_size = tf.placeholder(tf.int32, shape=())
            
            # Transmitter
            s = tf.random_uniform(shape=[batch_size], minval=0, maxval=self.M, dtype=tf.int64)
            x = self.encoder(s)     
                        
            # the attack vector
            p = tf.placeholder(tf.float32,shape=(None,2,self.n)) 
                        
            # Channel
            noise_std = tf.placeholder(tf.float32, shape=()) 
            noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=noise_std, seed=self.seed)
            y = x + noise + p
            
            # Receiver
            s_hat = self.decoder(y)
            
            
            # Loss function
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=s, logits=s_hat)
            
            
            # Performance metrics
            correct_predictions = tf.equal(tf.argmax(tf.nn.softmax(s_hat), axis=1), s)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            bler = 1-accuracy
            
            
            # Optimizer
            lr = tf.placeholder(tf.float32, shape=()) #
            train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        
            # References to graph variables we need to access later 
            self.vars = {
                'accuracy': accuracy,
                'batch_size': batch_size,
                'bler': bler,
                'cross_entropy': cross_entropy,
                'init': tf.global_variables_initializer(),
                'lr': lr,
                'noise_std': noise_std,
                'noise': noise,
                'p': p,
                's': s,
                's_hat': s_hat,
                'train_op': train_op,
                'x': x,
                'y': y,
            }            
            self.saver = tf.train.Saver()
        return
    
    def create_session(self):
        '''Create a session for the autoencoder instance with the compuational graph'''
        self.sess = tf.Session(graph=self.graph) # this is how we load the exact graph of the object instance       
        self.sess.run(self.vars['init'])
        return
    
    def encoder(self, input):
        '''The transmitter'''
        W = self.weight_variable((self.M,self.M))
        x = tf.nn.elu(tf.nn.embedding_lookup(W, input)) 
        x = tf.layers.dense(x, 2*self.n, activation=None)
        x = tf.reshape(x, shape=[-1,2,self.n]) 
        #Average power normalization
        x = x/tf.sqrt(2*tf.reduce_mean(tf.square(x))) 
        return x
    
    def decoder(self, input):
        '''The Receiver'''
        y = tf.reshape(input, shape=[-1,2*self.n])
        y = tf.layers.dense(y, 4*self.n, activation=tf.nn.relu)
        y = tf.layers.dense(y, self.M, activation=None)
        return y
       
    def EbNo2Sigma(self, ebnodb):
        '''Convert Eb/No in dB to noise standard deviation'''
        ebno = 10**(ebnodb/10)
        return 1/np.sqrt(2*self.bits_per_symbol*ebno) 
    
    def gen_feed_dict(self, perturbation, batch_size, ebnodb, lr):
        '''Generate a feed dictionary for training and validation'''        
        return {
            self.vars['p']: perturbation,
            self.vars['batch_size']: batch_size,
            self.vars['noise_std']: self.EbNo2Sigma(ebnodb),
            self.vars['lr']: lr}       
    
    def load(self, filename):
        '''Load a pre_trained model'''
        return self.saver.restore(self.sess, filename)
        
    def save(self, filename):
        '''Save the current model'''
        return self.saver.save(self.sess, filename)  
    
    def test_step(self, p, batch_size, ebnodb):
        '''Compute the BLER over a single batch and Eb/No'''
        bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr=0))
        return bler
    
    def transmit(self, s):
        '''Returns the transmitted sigals corresponding to message indices'''
        return self.sess.run(self.vars['x'], feed_dict={self.vars['s']: s})
       
    def train(self, p, training_params, validation_params):  
        '''Training and validation loop'''
        for index, params in enumerate(training_params):            
            batch_size, lr, ebnodb, iterations = params            
            print('\nBatch Size: ' + str(batch_size) +
                  ', Learning Rate: ' + str(lr) +
                  ', EbNodB: ' + str(ebnodb) +
                  ', Iterations: ' + str(iterations))
            
            val_size, val_ebnodb, val_steps = validation_params[index]
            
            for i in range(iterations):
                self.train_step(p, batch_size, ebnodb, lr)    
                if (i%val_steps==0):
                    bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(p,val_size, val_ebnodb, lr))
                    print(bler)                           
        return       
    
    def train_step(self, p, batch_size, ebnodb, lr):
        '''A single training step'''
        self.sess.run(self.vars['train_op'], feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr)) 
        return 
    
    def weight_variable(self, shape):
        '''Xavier-initialized weights optimized for ReLU Activations'''
        (fan_in, fan_out) = shape
        low = np.sqrt(6.0/(fan_in + fan_out)) 
        high = -np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    

    def bler_sim_attack_AWGN(self, p, PSR_dB, ebnodbs, batch_size, iterations):
        np.random.seed(seed=self.seed)
        PSR = 10**(PSR_dB/10)
        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p)**2 +  0.00000001) ) # 
        p = scale_factor * p
        BLER_no_attack = np.zeros_like(ebnodbs)
        BLER_attack = np.zeros_like(ebnodbs)
        BLER_jamming = np.zeros_like(ebnodbs)
        for i in range(iterations):
            # No attack - clean case
            bler = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(np.zeros([1,2,self.n]), batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) #
            BLER_no_attack = BLER_no_attack + bler/iterations
            # attack  - synchronous attack
            bler_attack = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(  p ,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # 
            BLER_attack = BLER_attack + bler_attack/iterations
            # Jamming attack
            normal_noise_as_jammer = np.random.normal(0,1,p.shape)
            jamming = np.linalg.norm(p) * (1 / np.linalg.norm(normal_noise_as_jammer)) * normal_noise_as_jammer
            bler_jamming= np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(jamming,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) 
            BLER_jamming = BLER_jamming + bler_jamming/iterations
        return BLER_no_attack, BLER_attack, BLER_jamming
   

    def fgm_attack(self,s,p, ebnodb): 
        '''Create an input specific adversarial example using the method proposed by Sadeghi and Larsson in [2] '''
        np.random.seed(seed=self.seed)
        num_class = self.M 
        y_reshaped = np.reshape(self.sess.run(self.vars['y'], feed_dict={self.vars['s']: s, self.vars['p']: p, self.vars['noise_std']: self.EbNo2Sigma(ebnodb)}), [1,2,self.n])         
        eps_acc = 0.0000001 * np.linalg.norm(y_reshaped)
        epsilon_vector = np.zeros([num_class])
        predictions = tf.nn.softmax(self.vars['s_hat'], name = 'predictions')        
        
        for cls in range(num_class):
            s_target = np.array([cls])
            adv_per_needtoreshape  = -1 * np.asarray( self.sess.run(tf.gradients(self.vars['cross_entropy'],self.vars['y']), feed_dict={self.vars['y']: y_reshaped , self.vars['s']:s_target }) )
            adv_per = adv_per_needtoreshape.reshape(1,2,self.n)
            norm_adv_per = adv_per / (np.linalg.norm(adv_per) +  0.000000000001)
            epsilon_max = 1 * np.linalg.norm(y_reshaped)
            epsilon_min = 0
            num_iter = 0
            wcount = 0
            while (epsilon_max-epsilon_min > eps_acc) and (num_iter < 30):
                wcount = wcount+1
                num_iter = num_iter +1
                epsilon = (epsilon_max + epsilon_min)/2
                adv_img_givencls = y_reshaped + (epsilon * norm_adv_per)
                
                predicted_probabilities = self.sess.run(predictions, feed_dict={self.vars['y']: adv_img_givencls})
            
                compare = np.equal(np.argmax(predicted_probabilities),s)
                if compare:
                    epsilon_min = epsilon
                else:
                    epsilon_max = epsilon
            epsilon_vector[cls] = epsilon + eps_acc
        false_cls = np.argmin(epsilon_vector)
        minimum_epsilon = np.min(epsilon_vector)
        adv_dirc = -1 * np.asarray(self.sess.run(tf.gradients(self.vars['cross_entropy'],self.vars['y']), feed_dict={self.vars['y']: y_reshaped, self.vars['s']: np.asarray([false_cls]) })  ).reshape(1,2,self.n)
        norm_adv_dirc = adv_dirc / (np.linalg.norm(adv_dirc) + 0.000000000001)
        adv_perturbation = minimum_epsilon * norm_adv_dirc
        return adv_perturbation, false_cls, minimum_epsilon
        

    
    def UAPattack_fgm(self,ebnodb,num_samples,PSR_dB):
        '''Create a Universal Adversarial Perturbation as suggested by Alg. 1 of Sadeghi et al in [2]'''
        np.random.seed(seed=self.seed)
        universal_per_fgm = np.zeros([1,2,self.n])
        for cnr_index in range(num_samples):#            
            s =  np.asarray([np.floor(np.random.uniform(0,16,1))]).reshape(1) 
            predicted_label = np.argmax( self.sess.run(self.vars['s_hat'], feed_dict={self.vars['s']:s, self.vars['p']:universal_per_fgm, self.vars['noise_std']: self.EbNo2Sigma(ebnodb)}) )
            if predicted_label == s:
                # First we need to find adverssarial direction for this instant  by solving eq. (1) of the paper
                adv_perturbation,_,_ = self.fgm_attack(s, universal_per_fgm,ebnodb)
                adv_perturbn_reshaped = adv_perturbation.reshape([1,2,self.n])
                UAP = universal_per_fgm + adv_perturbation.reshape([1,2,self.n])
                PSR = 10**(PSR_dB/10)
                Epsilon_uni = np.sqrt( (PSR * self.n) / (np.linalg.norm(UAP)**2 + 0.00000001) )
                # Second we need to revise the universal perturbation
                if np.linalg.norm(universal_per_fgm + adv_perturbn_reshaped) < Epsilon_uni: 
                    universal_per_fgm = universal_per_fgm + adv_perturbn_reshaped
                else:
                    universal_per_fgm =  Epsilon_uni * (universal_per_fgm + adv_perturbn_reshaped) 
        return universal_per_fgm


