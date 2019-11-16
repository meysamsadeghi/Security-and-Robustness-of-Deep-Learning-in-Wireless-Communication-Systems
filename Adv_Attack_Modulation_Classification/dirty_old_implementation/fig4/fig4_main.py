#Open saved files and plot fig4
import matplotlib.pyplot as plt
import pickle
import numpy as np


with open('/home/meysam/Work/ModulationClassification/codes_letter/fig3/PSR_SNR10_complete.pkl','rb') as g:#/home/meysam/ML/ModulationClassification/codes_letter/fig3
    [acc_moosavi_10,acc_noise_10,acc_grad_un_10,acc_grad_n_10,time_moosavi_10,time_gradnorm_10,Grad_Opt_10] = pickle.load(g)

with open('fig4_data','rb') as f:
    [PSRvec,ACC_BB_10,acc_grad_n_10,ACC_BB_0,acc_grad_n_0] = pickle.load(f)
    


#==============================================================================
fig, ax = plt.subplots()

ax.plot(PSRvec,75 * np.ones([11,1]),'ms--',label='No attack - SNR=10 dB')  


ax.plot(PSRvec,100 * acc_noise_10[0:11],'ko-',label='Jamming attack - SNR=10 dB')
 
ax.plot(PSRvec,100 * ACC_BB_10,'r^-',label='Black-box attack with random shifts - SNR=10 dB')        
ax.plot(PSRvec,100 * acc_grad_n_10[:11],'b>-',label='White-box attack of Alg. 2 - SNR=10 dB')




plt.legend(loc='lower left')
plt.xticks(PSRvec,PSRvec)

ax.set_xlabel('PSR [dB]')
ax.set_ylabel('Accuracy %')

ax.grid(True)
plt.savefig("fig4.eps",format='eps', dpi=1000)
plt.show()  