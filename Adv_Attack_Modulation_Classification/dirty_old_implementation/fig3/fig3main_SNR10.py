import numpy as np
import pickle
from fig3_function_without_importing import acc_psr_fun
import matplotlib.pyplot as plt

# initialize parameters
PSR_vec =  np.arange(-20,1,1)
print(PSR_vec)



num_times = 5 # number of Monte Carlo
N = 50 # number of samples
SNR = 10

l = len(PSR_vec)
acc_moosavi = np.zeros([l,1])
acc_noise = np.zeros([l,1])
acc_grad_un = np.zeros([l,1])
acc_grad_n = np.zeros([l,1])


time_moosavi = np.zeros([l,1])
time_gradnorm = np.zeros([l,1])

Acc_grad_n_optimal = 1
Grad_Opt = np.zeros([256,l])

for icn in range(len(PSR_vec)):
    PSR = PSR_vec[icn]

    acc_moosavi[icn,0],acc_noise[icn,0],acc_grad_un[icn,0],acc_grad_n[icn,0],time_moosavi[icn,0],time_gradnorm[icn,0],Grad_Opt[:,icn] , _ = acc_psr_fun(PSR,SNR,num_times,N)
    print('time_moosavi',time_moosavi[icn,0])
    print('time_gradnorm',time_gradnorm[icn,0])
    
#==============================================================================
     if icn == 11:
         fig, ax = plt.subplots()
         #ax.plot(PSR_vec[:11],acc_noise[:11],'ro-')
         ax.plot(PSR_vec[:11],acc_moosavi[:11],'r^-')
         ax.plot(PSR_vec[:11],acc_grad_n[:11],'b>-')
         #ax.plot(PSR_vec[:11],acc_grad_un[:11],'cs-')
         plt.xticks(np.arange(-20,-9,1),np.arange(-20,-9,1))
         ax.set_xlabel('PSR [dB]')
         ax.set_ylabel('Accuracy')
         ax.grid(True)
         plt.savefig("fig3_SNR10.png")
         plt.show()
#==============================================================================
            
# =============================================================================
#     with open('PSR_SNR10_GPU.pkl', 'wb') as f:
#         pickle.dump([acc_moosavi,acc_noise,acc_grad_un,acc_grad_n,time_moosavi,time_gradnorm,Grad_Opt], f)
# 
# =============================================================================

    





