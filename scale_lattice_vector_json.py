# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 16:47:37 2025

Script to scale c-vector of crystals in 2d-matpedia dataset by 1000.
Purpose of this is that atoms in adjacent layers of 2d crystals are no longer
seen as nearest neighbour atoms, in order to consider only one sheet.
"""

import json
import numpy as np
import os

origin_folder_path = "C:/Users/reise/Documents/Uni/Y4/MPhys_Project/trial_cgcnn/structure_database"
target_folder_path = "C:/Users/reise/Documents/Uni/Y4/MPhys_Project/trial_cgcnn/structure_database_adj"

SCALE_FACTOR = 1000

#get list of file names in folder
dir_index = os.listdir(origin_folder_path)

#iterate over files
for f in dir_index: 
    #open file
    with open(origin_folder_path + '/' + f, 'r') as file:
        data = file.read()
    
    crystal = json.loads(data) #convert string to dictionary
    #get c-vector, scale
    vector = SCALE_FACTOR* np.array(crystal['lattice']['matrix'][2])
    crystal['lattice']['matrix'][2] = [vector[0], vector[1], vector[2]]
    
    #save as OldFileNam_adj.json
    with open(target_folder_path + '/' + f.split('.')[0] + '_adj.json', "w") as g:
        json.dump(crystal, g)
