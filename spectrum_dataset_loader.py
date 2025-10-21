# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 16:13:18 2025

@author: reise
"""


from torch.utils.data import Dataset
import os
import numpy as np

class SpecDsLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        self.ids = [ele for ele in os.listdir(self.root_dir)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        spectrum = np.genfromtxt(self.root_dir + '/' + self.ids[idx], skip_header=1, delimiter=',', dtype=float)
        return spectrum