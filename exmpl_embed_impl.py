# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 16:18:50 2025

@author: reise
"""

import argparse
import torch
import torch.nn as nn
from json_data_loader import JSONData

class arguments:
    def __init__(self):
        self.modelpath = "C:/Users/reise/Desktop/Raman_spectra/CGCNN/pre-trained/band-gap.pth.tar"
        self.root_dir = "C:/Users/reise/Documents/Uni/Y4/MPhys_Project"
        self.batch_size = 256
        self.workers = 0
        self.print_freq = 10
        self.disable_cuda = True
        
args = arguments()

model_checkpoint = torch.load(args.modelpath,
                                map_location=lambda storage, loc: storage)
weights = model_checkpoint['state_dict']['embedding.weight']
biases = model_checkpoint['state_dict']['embedding.bias']
model_args = argparse.Namespace(**model_checkpoint['args'])


dataset = JSONData(args.root_dir, '2dm-1')
structures = dataset[0]
orig_atom_fea_len = structures[0].shape[-1]
nbr_fea_len = structures[1].shape[-1]

embedder = nn.Linear(orig_atom_fea_len, model_args.atom_fea_len)
embedder.weight.data = weights
embedder.bias.data = biases

print(embedder(structures[0]))