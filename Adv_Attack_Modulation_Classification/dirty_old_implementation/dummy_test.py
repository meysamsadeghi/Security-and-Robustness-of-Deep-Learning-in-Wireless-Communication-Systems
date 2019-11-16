import pickle
import numpy as np


with open('NR10PNR0.pkl', 'rb') as f:
    [acc_moosavi, acc_pcamax_n, acc_noise,acc_grad_un,acc_grad_n,acc_sgl_img]= pickle.load(f)
