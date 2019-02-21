# hamming coding

import numpy as np

def hamming_74(n,k,ebnodbs,num_blocks,p,PSR_dB):
    PSR = 10**(PSR_dB/10)
    preal = p[:7] 
    
    
    SNR_dB = ebnodbs  # I think SNR and Eb/No are the same given the way I used them
    SNR = 10**(SNR_dB/10)
    SNR_Hamming = SNR*(k/n)
    #
    data_bits = np.floor(np.random.uniform(0,2,[k,num_blocks]))
    data_bits = data_bits.astype(int)
    parity1 = ( data_bits[0,:] ^ (data_bits[1,:] ^ data_bits[3,:]) )
    parity2 = ( data_bits[0,:] ^ (data_bits[2,:] ^ data_bits[3,:]) )
    parity3 = ( data_bits[1,:] ^ (data_bits[2,:] ^ data_bits[3,:]) )
    Hamming_Code = np.vstack( (data_bits,parity1,parity2,parity3) )
    #
    bpsk = -1 * np.ndarray.astype( (Hamming_Code==0)  , int) + Hamming_Code
    #
    H = np.array([[1, 1, 0, 1, 1, 0, 0],[1, 0, 1, 1, 0, 1, 0],[0 , 1, 1, 1, 0, 0 ,1]])
    #P = H[:,:k].T 
    #
    messages= np.array([[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])
    messages_p1 = ( messages[0,:] ^ (messages[1,:] ^ messages[3,:]) )
    messages_p2 = ( messages[0,:] ^ (messages[2,:] ^ messages[3,:]) )
    messages_p3 = ( messages[1,:] ^ (messages[2,:] ^ messages[3,:]) )
    messages = np.vstack( (messages,messages_p1,messages_p2,messages_p3) )
    messages_after_bpsk = -1 * np.ndarray.astype( (messages==0)  , int) + messages
    
    
    #----------------- MonteCarlo Implementation ---------------------------------
    BLER_HD = np.zeros(len(SNR_dB)) 
    BLER_HD_adv = np.zeros(len(SNR_dB))
    BLER_HD_jam = np.zeros(len(SNR_dB)) 
    y_decoded_HD_adv = np.zeros([n,num_blocks]) 
    y_decoded_HD_jam = np.zeros([n,num_blocks])
    y_decoded_HD = np.zeros([n,num_blocks]) 
    
    BLER_MLD = np.zeros(len(SNR_dB))
    BLER_MLD_adv = np.zeros(len(SNR_dB))
    BLER_MLD_jam = np.zeros(len(SNR_dB))
    y_decoded_MLD = np.zeros([n,num_blocks])
    y_decoded_MLD_adv = np.zeros([n,num_blocks])
    y_decoded_MLD_jam = np.zeros([n,num_blocks])
    for snr_cntr in range(len(SNR)):
        # No attack
        y = np.sqrt(2*SNR_Hamming[snr_cntr]) * bpsk + ( np.random.normal(0,1,[n,num_blocks]) )  
        # Adversarial attack
        y_adv = bpsk + ( (1/np.sqrt(2*SNR_Hamming[snr_cntr]) ) * np.random.normal(0,1,[n,num_blocks]) ) + (  (np.sqrt(n * PSR /2)  * (preal / np.linalg.norm(preal))  ).reshape([n,1]) * np.ones([n,num_blocks]) )
        
        # Jamming Attack
        jammer_noise = np.random.normal(0,1,[n,num_blocks])
        y_jam = bpsk + ( (1/np.sqrt(2*SNR_Hamming[snr_cntr]) ) * np.random.normal(0,1,[n,num_blocks]) ) + ( np.sqrt(n * PSR / 2 ) * (1/np.linalg.norm(jammer_noise,axis=0)) * jammer_noise )
        # Hard Decision Decoding
        y_binary  = np.ndarray.astype( (y>=0)  , int)  #  tarnsfering -1 and 1 into 0 and 1 <==> hard decoding
        y_decoded_HD = y_binary
        y_binary_adv  = np.ndarray.astype( (y_adv>=0)  , int)  #  tarnsfering -1 and 1 into 0 and 1 <==> hard decoding
        y_binary_jam  = np.ndarray.astype( (y_jam>=0)  , int)
        y_decoded_HD_adv = y_binary_adv
        y_decoded_HD_jam = y_binary_jam
        for blk_cntr in range(num_blocks):
            syndrom = np.matmul(H , y_binary[:,blk_cntr]) % 2
            syndrom_adv = np.matmul(H , y_binary_adv[:,blk_cntr]) % 2
            syndrom_jam = np.matmul(H , y_binary_jam[:,blk_cntr]) % 2 
            for bit_cntr in range(n):
                if np.array_equal(H[:,bit_cntr],syndrom):
                    y_decoded_HD[bit_cntr,blk_cntr] = y_binary[bit_cntr,blk_cntr] ^ 1
                if np.array_equal(H[:,bit_cntr],syndrom_adv):
                    y_decoded_HD_adv[bit_cntr,blk_cntr] = y_binary_adv[bit_cntr,blk_cntr] ^ 1 # HERE we flip (0 to 1 and vice versa) bit_cntr-th bit of blk_cntr-th sample out of the num_bolck
                if np.array_equal(H[:,bit_cntr],syndrom_jam):
                    y_decoded_HD_jam[bit_cntr,blk_cntr] = y_binary_jam[bit_cntr,blk_cntr] ^ 1  
            if np.sum(np.ndarray.astype( np.not_equal(y_decoded_HD[:4,blk_cntr],data_bits[:,blk_cntr]), int)) > 0:
                BLER_HD[snr_cntr] = BLER_HD[snr_cntr] + (1 / num_blocks)
            if np.sum(np.ndarray.astype( np.not_equal(y_decoded_HD_adv[:4,blk_cntr],data_bits[:,blk_cntr]), int)) > 0:
                BLER_HD_adv[snr_cntr] = BLER_HD_adv[snr_cntr] + (1 / num_blocks)
            if np.sum(np.ndarray.astype( np.not_equal(y_decoded_HD_jam[:4,blk_cntr],data_bits[:,blk_cntr]), int)) > 0:
                BLER_HD_jam[snr_cntr] = BLER_HD_jam[snr_cntr] + (1 / num_blocks)
    
        # Maximum Likelihood Decoding (MLD)
        for blk_cntr in range(num_blocks):
            # no attack
            distance = np.zeros([2**k])
            for clmn_cntr in range(2**k):
                distance[clmn_cntr] = np.linalg.norm(messages_after_bpsk[:,clmn_cntr] - y[:,blk_cntr]) 
            y_decoded_MLD[:,blk_cntr] = messages[:, np.argmin(distance)]
            if np.sum(np.ndarray.astype( np.not_equal(y_decoded_MLD[:4,blk_cntr],data_bits[:,blk_cntr]), int)) > 0:
                BLER_MLD[snr_cntr] = BLER_MLD[snr_cntr] + (1 / num_blocks)
            # adversarial
            distance_adv = np.zeros([2**k])
            for clmn_cntr in range(2**k):
                distance_adv[clmn_cntr] = np.linalg.norm(messages_after_bpsk[:,clmn_cntr] - y_adv[:,blk_cntr]) # 
            y_decoded_MLD_adv[:,blk_cntr] = messages[:, np.argmin(distance_adv)]
            if np.sum(np.ndarray.astype( np.not_equal(y_decoded_MLD_adv[:4,blk_cntr],data_bits[:,blk_cntr]), int)) > 0:
                BLER_MLD_adv[snr_cntr] = BLER_MLD_adv[snr_cntr] + (1 / num_blocks)
            # jamming
            distance_jam = np.zeros([2**k])
            for clmn_cntr in range(2**k):
                distance_jam[clmn_cntr] = np.linalg.norm(messages_after_bpsk[:,clmn_cntr] - y_jam[:,blk_cntr]) # 
            y_decoded_MLD_jam[:,blk_cntr] = messages[:, np.argmin(distance_jam)]
            if np.sum(np.ndarray.astype( np.not_equal(y_decoded_MLD_jam[:4,blk_cntr],data_bits[:,blk_cntr]), int)) > 0:
                BLER_MLD_jam[snr_cntr] = BLER_MLD_jam[snr_cntr] + (1 / num_blocks)

    return BLER_HD, BLER_MLD, BLER_HD_adv, BLER_MLD_adv, BLER_HD_jam, BLER_MLD_jam














































































