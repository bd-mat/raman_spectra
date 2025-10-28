# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:33:36 2025

@author: reise
"""

import numpy as np
from get_raman_spectra import GetRaman

n_peaks = 4

directory = "C:/Users/reise/Documents/Uni/Y4/MPhys_Project/Li_data/dataset"
dataset = GetRaman(directory, n_peaks=n_peaks)

peak_data_all_cryst = dataset.get_many_spec(dataset.id_num_list)

output_arr = np.empty((0,2))
for idx, id_num in enumerate(dataset.id_num_list):

    line = np.array([dataset.dir_index[id_num],  peak_data_all_cryst[:,:,idx].flatten()], dtype=object) # output eigenfreq and int
    #line = np.array([dataset.dir_index[id_num], eig_freqs_all_cryst[:,idx]], dtype=object)  # output only eigenfreq
    output_arr = np.vstack((output_arr, line))
        
    
    
filename = "C:/Users/reise/Documents/Uni/Y4/MPhys_Project/Li_data/id_prop"
header = 'ID Number, 4 strongest eigenfrequency [cm^-1]'
#np.savetxt(filename, output_arr, delimiter=',', fmt='%.0f %.5f', header=header)
np.save(filename, output_arr, allow_pickle=True)