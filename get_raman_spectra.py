# -*- coding: utf-8 -*-
"""
Extract the Raman spectrum information from the dataset created by 
Li et. al. 2025

written 10-10-2025
"""

import numpy as np
import os

class GetRaman:
    """
    Get raman spectra from the folders.
    
    Assumed folder structure:
        root_dir
        ├── 1_ChemFormula
            ├── aman.dat
        ├── 2_ChemFormula
            ├── aman.dat
        ├── 3_ChemFormula
            ├── aman.dat

    
    """
    
    def __init__(self, folder_path, n_peaks=10):
        self.root = folder_path
        self.n_peaks = n_peaks
        
        #index the database folder
        assert os.path.exists(self.root), f'ERROR: {self.root} does not exist'
        #   create dictionary {crystal_id, folder name}
        self.dir_index = {f"{ele.split('_')[0]}": ele for ele in os.listdir(self.root)}
        
    def get_single_spec(self, crystal_id):
        """
        Retrieves the entire raman spectrum from the subfolder in the root
        directory and outputs as array.

        Parameters
        ----------
        crystal_id : int
            number assigned to crystal in database

        Returns
        -------
        raman_spectrum : array
            2d numpy array in the form
            (frequency [cm^-1], intensity [Angstrom^-4/amu])

        """
        raman_spectrum = np.empty((0,2)) #initialise output array
        name = self.dir_index[str(crystal_id)]  #get folder name for id
        fpath = self.root + f'/{name}' + '/raman.dat'
        with open(fpath, 'r') as file:
            for line in file: #iterate over raman peaks and place into array
                line = line.replace("\n", "")
                line = np.array(line.split(' ')).astype(float)
                raman_spectrum = np.vstack((raman_spectrum, line))
        return raman_spectrum
    
    def equalise_length(self, raman_spectrum):
        #if there are more peaks than needed, take the strongest
        if len(raman_spectrum[:,0]) > self.n_peaks:
            #sort spectrum by intensity
            raman_spectrum = raman_spectrum[np.flip(np.argsort(raman_spectrum[:,1])),:]
            #clip beyond npeaks
            raman_spectrum = raman_spectrum[:self.n_peaks,:]
            
        #if there are fewer peaks than needed than fill with zeros
        elif len(raman_spectrum) < self.n_peaks:
            zeros = np.zeros((self.n_peaks-len(raman_spectrum), 2)) #array of zeros
            raman_spectrum = np.vstack((raman_spectrum, zeros)) #concatenate arrays
        
        return raman_spectrum
    
    def get_many_spec(self, crystal_id_list):
        """
        Gets the raman spectra for a list of crystals from the dataset.
        The number of raman peaks is equalised for all crystals using the
        equalise_length method.

        Parameters
        ----------
        crystal_id_list : list
            list of unique crystal identifier numbers that the raman spectrum
            is to be retrieved for

        Returns
        -------
        raman_spectra : numpy array
            3d numpy array containing the raman spectrum for the crystals
            format: (eigen frequency, intensity, crystal)

        """
        #initialise array
        raman_spectra = np.empty((self.n_peaks,2,0))
        for cryst_id in crystal_id_list:
            raman_spectrum = self.get_single_spec(cryst_id)
            raman_spectrum = self.equalise_length(raman_spectrum)
            raman_spectra = np.dstack((raman_spectra, raman_spectrum))
        return raman_spectra
    