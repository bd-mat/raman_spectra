# -*- coding: utf-8 -*-

"""
Created on Thu Oct 30 13:32:43 2025

@author: reise
"""

import numpy as np
import matplotlib.pyplot as plt
from eig_hyps import trainModel
import time
from tqdm import tqdm
from matplotlib import cm

folder = "C:/Users/reise/Documents/Uni/Y4/MPhys_Project/big_NN/training_set"

Nk1_vals = 6
Nk2_vals =6
k1 = np.linspace(0.4,0.85, Nk1_vals)
k2 = np.linspace(0.9,3, Nk1_vals)

def iterator2d(vals1, vals2, folder, vals1_name, vals2_name, 
               MAE_decrease_plot=False, **kwargs):
    
    results = np.zeros((len(vals1), len(vals2)))
    for i, numi in enumerate(tqdm(vals1, desc='vals1')):
        for j, numj in enumerate(tqdm(vals2, desc='vals2')):
            run_kwargs = kwargs.copy()
            run_kwargs[vals1_name] = numi
            run_kwargs[vals2_name] = numj
            
            MAEs, loss, hyperparams = trainModel(**run_kwargs)
            print(hyperparams)
            
            if MAE_decrease_plot == True:
                plot_training(MAEs, hyperparams, folder)
            print(f'MAE = {MAEs[-1]:.3f}')
            results[i,j] = MAEs[-1]
    print('Iterations complete')
    plot_surface(vals1, vals2, results, vals1_name, vals2_name)
    return results

def plot_training(av_mae_arr, parameters, folder):
   n_h, k1, k2 = parameters[0], parameters[1], parameters[2]
   plt.plot(av_mae_arr)
   plt.ylabel('Average MAE')
   plt.xlabel('Epoch')

   text = f' k1={k1}, k2={k2}\n n_h={n_h}\n MAE={av_mae_arr[-1]:.3f}'
   ax = plt.gca()
   plt.text(0.7, 0.8, text, transform = ax.transAxes)
   fname = folder + f'/{k1}-{k2}-{n_h}.jpg'
   plt.savefig(fname, dpi = 150, bbox_inches='tight')
   plt.show()
   return None

def plot_surface(vals1, vals2, results, vals1_name, vals2_name):
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(vals1, vals2)
    surf = ax.contourf(X, Y, results, cmap=cm.magma,
                       linewidth=0, antialiased=False)
    ax.set_xlabel(vals1_name)
    ax.set_ylabel(vals2_name)
    
    fig.colorbar(surf, ax=ax)
    plt.show()
    return None

results = iterator2d(k1, k2, folder, 'k1', 'k2', MAE_decrease_plot=True, 
                     n_h = 20, EPOCHS=30, lr=0.1)



        




