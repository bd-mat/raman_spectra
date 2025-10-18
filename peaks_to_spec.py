# -*- coding: utf-8 -*-
"""
Tool script to calculate and view the raman spectras based on a list of peaks
and intensities.

Created on Thu Oct 16 15:39:30 2025

"""

import matplotlib.pyplot as plt
import numpy as np
from get_raman_spectra import GetRaman

def gaussian(x, A, sig, mu):
    """
    Gaussian function

    Parameters
    ----------
    x : float
        argument
    A : float
        height at peak
    sig : float
        standard deviation
    mu : float
        location of peak

    Returns
    -------
    float

    """
    return A*np.exp(-((x-mu)**2) /(2*sig**2))

def calc_spec(x, raman_peaks, peak_sig):
    """
    Converts a list of peaks into a spectral representation y(x), by inserting
    a gaussian around each peak listed in the raman peaks array. The height of
    the gaussian is the intensity of the peak.
    Unit of x has to be same as unit of raman_spec[:,0].

    Parameters
    ----------
    x : 1d array
        DESCRIPTION.
    raman_peaks : 2d array
        List if peak frequencies and their intensity.
    peak_sig : float
        Width of gaussian peak.
    Returns
    -------
    y : 1d array
        Intensity, in same unit as raman_peaks[:,1]

    """
    y = np.zeros(len(x))

    for i in range(len(raman_peaks[:,0])):
        y += gaussian(x, raman_peaks[i,1], peak_sig, raman_peaks[i,0])
    
    return y


SIG = 15 #sigma for gaussian
SOURCE_FOLDER = "C:/Users/reise/Documents/Uni/Y4/MPhys_Project/Li_data/data"
TARGET_FOLDER = "C:/Users/reise/Documents/Uni/Y4/MPhys_Project/Li_data/spectra/"

#genearte list of frequency values
LEN = 3000
fmin = 0 #min(rs[:,0])
fmax = 4410 #max(rs[:,0])
x = np.linspace(fmin, fmax, LEN) #[cm^-1]

#initialise class to get raman spectra
rs_database = GetRaman(SOURCE_FOLDER)
id_num_list = rs_database.id_num_list #dataset ids
dir_index = rs_database.dir_index #names of folders in dataset

#make sorted list of id numbers
id_num_list = np.array(id_num_list).astype(int)
id_num_list = np.sort(id_num_list) 

#Investigate single 
INDEX = 1 #index in list of dataset id numbers
#   get list of peaks
raman_peaks = rs_database.get_single_spec(id_num_list[INDEX]) 
#   calculate spectrum
y = calc_spec(x, raman_peaks, SIG)
print(raman_peaks)

#   plot
plt.plot(x,y)
plt.title(f"{dir_index[str(id_num_list[INDEX])].split('_')[1]}")
plt.xlabel('Frequency [$cm^1$]')
plt.ylabel('Intensity [$\AA^4$ / amu]')


#export intensity values for all
"""
for i in range(len(id_num_list)):
    rs = rs_database.get_single_spec(id_num_list[i])
    y = calc_spec(x, rs, SIG)
    
    fname = TARGET_FOLDER + rs_database.id_num_list[i] + '_spec.txt'
    header = 'Intensity [angstrom^4/amu]'
    np.savetxt(fname, y, delimiter=',', header=header)
"""


#Get frequency range
"""
for idx in rs_database.dir_index:
    spec_i = rs_database.get_single_spec(idx)
    min_i = min(spec_i[:,0])
    max_i = max(spec_i[:,0])
    if min_i < fmin:
        fmin = min_i
    if max_i > fmax:
        fmax = max_i
        
print(fmin, fmax)

"""




