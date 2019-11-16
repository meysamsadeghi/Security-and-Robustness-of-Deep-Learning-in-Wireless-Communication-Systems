# PLOT of Fig 3
import numpy as np
import matplotlib.pyplot as plt
import pickle

PSR_vec =  np.arange(-20,1,1)


[acc_moosavi_0,acc_noise_0,acc_grad_un_0,acc_grad_n_0,time_moosavi_0,time_gradnorm_0,Grad_Opt_0] = pickle.load(open('PSR_SNR0_complete.pkl','rb'))
[acc_moosavi_10,acc_noise_10,acc_grad_un_10,acc_grad_n_10,time_moosavi_10,time_gradnorm_10,Grad_Opt_10] = pickle.load(open('PSR_SNR10_complete.pkl','rb'))



fig, ax = plt.subplots()

#============ SNR=0 ===================
#==============================================================================
# ax.plot(PSR_vec[:11],acc_noise_0[:11],'gX-',label='Jammer - SNR=0 dB')
# ax.plot(PSR_vec[:11],acc_moosavi_0[:11],'m1-',label='UAP of [13] - SNR=0 dB')
# ax.plot(PSR_vec[:11],acc_grad_n_0[:11],'b2-',label='UAP Alg. 2 - SNR=0 dB')
# #ax.plot(PSR_vec[:11],acc_grad_un_0[:11],'cs-')
#==============================================================================
#============ SNR=10 ===================
ax.plot(PSR_vec[:11],75 * np.ones([11,1]),'ms--',label='No attack - SNR=10 dB')
ax.plot(PSR_vec[:11],100 * acc_noise_10[:11],'ko-',label='Jamming attack - SNR=10 dB')
ax.plot(PSR_vec[:11],100 * acc_moosavi_10[:11],'r^-',label='UAP attack of [Moosavi, 2017] - SNR=10 dB')
ax.plot(PSR_vec[:11],100 * acc_grad_n_10[:11],'b>-',label='UAP attack of Alg. 2 - SNR=10 dB')
#ax.plot(PSR_vec[:11],acc_grad_un_10[:11],'cs--')

plt.legend(loc='lower left')

plt.xticks(np.arange(-20,-9,1),np.arange(-20,-9,1))

ax.set_xlabel('PSR [dB]')
ax.set_ylabel('Accuracy %')

ax.grid(True)
plt.savefig("fig3_SNR10_pr.pdf",format='pdf', dpi=1000)
plt.show()