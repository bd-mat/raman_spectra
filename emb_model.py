# %%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset
import os

class SpecDsLoader(Dataset):
    def __init__(self, root_dir):
        super(SpecDsLoader,self).__init__()
        self.root_dir = root_dir
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        self.ids = [ele for ele in os.listdir(self.root_dir)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        spectrum = np.genfromtxt(self.root_dir + '/' + self.ids[idx], skip_header=1, delimiter=',', dtype=float)
        return spectrum

class encoder(nn.Module):
    """
    Encodes an input raman spectrum via a bottleneck, and then
    decodes the result.
    """
    def __init__(self,spec_length=3000,encode_length=300,n_hidden=1,h_length=1000):
        """
        Initialise the encoder.

        Parameters
        ----------

        spec_length: int
          Length of the input spectrum.
        encode_length: int
          Length of the encoded vector.
        """
        super(encoder,self).__init__()
        # Parameters
        #self.activate = nn.Softplus()
        self.activate = nn.ReLU()
        self.spec_length = spec_length
        self.encode_length = encode_length
        self.n_hidden = n_hidden
        self.h_length = h_length
        # Layers
        if self.n_hidden == 0:
            self.encode = nn.Linear(self.spec_length,self.encode_length) # encoding layer
            self.decode = nn.Linear(self.encode_length,self.spec_length) # decoding layer
        else:
            self.encode = nn.Linear(self.h_length,self.encode_length)
            self.decode = nn.Linear(self.encode_length,self.h_length)
        # hidden layers
        if self.n_hidden >= 1:
            # initial conversion
            self.h_init = nn.Linear(spec_length,h_length)
            self.h_fin = nn.Linear(h_length,spec_length)
            # connected layers
            if self.n_hidden>1:
                self.hs1 = nn.ModuleList([nn.Linear(h_length,h_length) for _ in range(n_hidden-1)])
                self.hs2 = nn.ModuleList([nn.Linear(h_length,h_length) for _ in range(n_hidden-1)])
    
    def forward(self,in_spectrum):
        ins = in_spectrum
        # optional hidden layers
        if self.n_hidden >= 1:
            hid = self.activate(self.h_init(ins))
            if self.n_hidden > 1:
                for h in self.hs1:
                    hid = self.activate(h(hid))
        else:
            hid = ins
        # encode
        enc = self.encode(hid)
        enc = self.activate(enc)
        # and now, decode
        hout = self.decode(enc)
        hout = self.activate(hout)
        # optional hidden layers
        if self.n_hidden >= 1:
            if self.n_hidden > 1:
                for h in self.hs1:
                    hout = self.activate(h(hout))
            out = self.activate(self.h_fin(hout))
        else:
            out = hout
        return out

# custom loss function
def weightedloss(w):
    def loss(output,target):
        dif = torch.square(target-output)
        #sum = torch.abs(target+output)
        comb = torch.mul(dif,dif)
    return loss
# %%

p_train = 0.6
p_val = 0.2
batch_size = 256
num_workers = 0
do_cuda = False
shuffle = True # shuffle indices
np.random.seed(1234)

EPOCHS = 10

root_dir = "C:/Users/bjama/Desktop/MPhys/spectrum_embedding/fake_spectra"

# %%

# loss function
loss_fn = nn.MSELoss()
# define our model
model = encoder(n_hidden=5,h_length=1000)
# optimise via sgd
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # maybe use 
# create dataloader
train_dataset = SpecDsLoader(root_dir) # TBC
valid_dataset = SpecDsLoader(root_dir) # TBC
N = len(train_dataset)
inds = list(range(N))
s1 = int(np.floor(p_train*N))
s2 = int(np.floor((p_train+p_val)*N))
if shuffle:
    np.random.shuffle(inds)
train_ids,valid_ids = inds[:s1],inds[s1:s2]
train_sampler = SubsetRandomSampler(train_ids)
valid_sampler = SubsetRandomSampler(valid_ids)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,sampler=train_sampler,num_workers=num_workers,pin_memory=do_cuda)
valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size,sampler=valid_sampler,num_workers=num_workers,pin_memory=do_cuda)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    for i,dats in enumerate(train_loader):
        ins = dats.to(torch.float32)
        optimizer.zero_grad()

        outputs = model(ins)
        loss = loss_fn(outputs,ins)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 5 == 4:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss
# %%
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}'.format(epoch_number+1))
    # train
    model.train(True)
    avg_loss = train_one_epoch(epoch_number,writer)

    running_vloss = 0.0
    # eval time
    model.eval()
    with torch.no_grad():
        for i,vdata in enumerate(valid_loader):
            vins = vdata.to(torch.float32)
            vouts = model(vins)
            vloss = loss_fn(vouts,vins)
            running_vloss += vloss
    
    avg_vloss = running_vloss / (i+1)
    print('LOSS train {} valid {}'.format(avg_loss,avg_vloss))
    # etc
    writer.add_scalars('Training vs. Validation Loss',{ 'Training' : avg_loss, 'Validation' : avg_vloss }, epoch_number + 1)
    writer.flush()
    # something else
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
# %%
