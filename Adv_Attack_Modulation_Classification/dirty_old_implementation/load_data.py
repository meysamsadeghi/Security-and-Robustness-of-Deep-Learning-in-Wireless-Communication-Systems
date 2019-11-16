




#==============================================================================
# The Modulation Classification Dataset by Tim Oshea
# It is GNU radio generated datset which accounts for practical aspect of wieless channels such as ...
# It 220,000 samples. Each sample is 2*128 matrix containing the IQ (2) of a received signal sampled 128 times.
# It contains sample points for 11 diifferent modulation schemes (BPSK,QAM,8PSK,...) at 20 different SNR levels, which adds up to 220 tuples of ('modulation',SNR) where for each tuple we have 1000 samples of dimenssion 2*128.



def ModCls_loaddata():
    import numpy as np
    import pickle as cPickle
    import sys
    sys.path.insert(0, '/home/meysam/Work/ModulationClassification/codes_letter')
    
    
    # There is a Pickle incompatibility of numpy arrays between Python 2 and 3
    # which generates ascii encoding error, to work around that we use the following instead of
    # Xd = cPickle.load(open("RML2016.10a_dict.dat",'rb'))
    with open('/home/meysam/Work/ModulationClassification/codes_letter/RML2016.10a_dict.dat','rb') as ff:
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
    classes = mods
    return(X,lbl,X_train,X_test,classes,snrs,mods,Y_train,Y_test,train_idx,test_idx)

