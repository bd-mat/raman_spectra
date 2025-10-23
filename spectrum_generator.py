# %%
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr

nr.seed(1234)
# %%

# we want to generate random spectra, which are well distributed over the frequency space
# start by generating 2-8 'eigenvalues' in range

def get_eigs():
    n = nr.randint(2,8)
    # make sure they are well separated
    while True:
        eigs = nr.randint(0,3000,size=(n,))
        if check_eigs(eigs):
            break
    ints = nr.uniform(0.1,1,size=(n,))
    return eigs,ints

def check_eigs(eigs):
    tol = 25
    for i in range(0,len(eigs)):
        for j in range(0,len(eigs)):
            if i != j and np.abs(eigs[i]-eigs[j]) <= tol:
                return False
    return True


def gaussian(x,mu=0,h=1,std=25):
    return h*np.exp(-0.5*(((x-mu)/(std))**2))
    

def convolve(eigs,ints):
    x = np.array(range(3000))
    y = np.zeros((3000,),dtype=np.float32)
    for i,j in zip(eigs,ints):
        y += gaussian(x,mu=i,h=j)
    return y

def get_spectrum():
    eigs,ints = get_eigs()
    ints = ints/np.max(ints)
    C = convolve(eigs,ints)
    return C
# %%
# Now we can make a *bunch* of these and output them to form a dataset
dir = "C:/Users/bjama/Desktop/MPhys/spectrum_embedding/fake_spectra"

for i in range(0,50000):
    spec = get_spectrum()
    np.savetxt(fname=dir+'/'+'spec'+str(i)+'.csv',X=spec,newline="\n",header="Spectral Intensity")