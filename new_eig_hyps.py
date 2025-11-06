import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from eigenfreq_dataset_loader import JSONData
from eigenfreq_dataset_loader import collate_pool
from eigenfreq_dataset_loader import get_train_val_test_loader
from torch.utils.data.dataloader import default_collate

from torch.utils.data import Dataset, DataLoader
import os

import eig_model as eg

def mae(output,target,ind):
    return torch.mean(torch.abs(target-output)[:,int(ind)])

def wloss2(k1,k2):
    def loss(output, target):
        diff = torch.abs(target-output)
        diff[:,:4] *= target[:,4:]*k1
        diff[:,4:] *= k2
        return torch.mean(diff)
    return loss

def freqloss():
    def loss(output,target):
        diff = torch.abs(target-output)
        return torch.mean(diff[0:4])

class Normalizer(object):
    def __init__(self,scale_fac):
        self.scale_fac = 1/scale_fac
        self.inv_fac = scale_fac

    def norm(self,tensor):
        temp = tensor
        temp[:,:4] *= self.scale_fac
        return temp
    
    def denorm(self,tensor):
        temp = tensor
        temp[:,:4] *= self.inv_fac
        return temp

class logNormalizer(object):
    def __init__(self,scale_fac):
        self.scale_fac = scale_fac

    def norm(self,tensor):
        temp = tensor
        temp[:,:4] = torch.log(torch.abs(temp[:,:4]))/np.log(self.scale_fac)
        return temp
    
    def denorm(self,tensor):
        temp = tensor
        temp[:,:4] = torch.exp(temp[:,:4]*np.log(self.scale_fac))
        return temp

def trainModel(
    lr=0.001,
    EPOCHS=30,
    p_train = 0.6,
    p_val = 0.2,
    p_test = 0.1,
    seed=123,
    n_h = 25,
    n_conv = 3,
    h_fea_len = 128,
    atom_fea_len = 64,
    root_dir = "C:/Users/bjama/Desktop/big_NN/big_NN/data",
    adam = False,
    infin = False,
    test = True,
    scale=4410,
    k1=1,
    k2=0.8,
    bsize=256,
    lognorm = False,
    res=False,
    n_r=3,
    n_r_fea=64,
    n_r_h_fea=32,
    testing=False,
    fname='id_prop.npy',
    lossmode=None,
    peaks=False,
    patience=10
):
    torch.manual_seed(seed)
    mom = 0.9
    wd = 0
    do_cuda = False
    batch_size = bsize
    num_workers = 0
    if lossmode==None and not peaks:
        loss_fn =  wloss2(k1,k2)
    elif lossmode=='L1Loss':
        loss_fn = nn.L1Loss()
    elif lossmode == 'MSELoss':
        loss_fn = nn.MSELoss()
    else:
        print('Invalid Loss Mode.')
        return (1,1,1)
    
    if not peaks:
        scale_fac = scale
    else:
        scale_fac = 1


    dataset = JSONData(root_dir,fname=fname)

    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    if peaks:
        outlen = 1
    else:
        outlen = 8

    if testing:
        model = eg.cat_CGCNN(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=atom_fea_len,
                                    n_conv=n_conv,
                                    n_h=n_h,
                                    h_fea_len=h_fea_len,output_len=outlen)
    elif res:
        model = eg.resnet_cgcnn(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=atom_fea_len,
                                    n_conv=n_conv,
                                    n_h=n_h,
                                    h_fea_len=h_fea_len,cgcnn_len=n_r_fea,
                                    n_r=n_r,r_hidden_fea=n_r_h_fea,output_len=outlen)
    else:
        model = eg.CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=atom_fea_len,
                                    n_conv=n_conv,
                                    n_h=n_h,
                                    h_fea_len=h_fea_len,output_len=outlen)
    


    if adam and infin:
        infin = False
    if adam:
        optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,"min",patience=patience,threshold=0.0003)
    if infin:
        EPOCHS = 200 # not quite "infinite", but you get the idea

    # make loaders
    collate_fn = collate_pool
    train_loader, valid_loader, _ = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        train_ratio=p_train,
        num_workers=num_workers,
        val_ratio=p_val,
        test_ratio=p_test,
        pin_memory=do_cuda,
        gen=None,
        train_size=None,
        val_size=None,
        test_size=None,
        return_test=True)
    # end
    if lognorm:
        normalizer = logNormalizer(scale_fac=scale_fac)
    else:
        normalizer = Normalizer(scale_fac=scale_fac)
    # epoch training
    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.
        for i,(ins,targs,_) in enumerate(train_loader):
            # load inputs
            input_var = (ins[0].requires_grad_(True),
                        ins[1].requires_grad_(True),
                        ins[2],
                        ins[3])
            # normalize targets
            targs_normed = normalizer.norm(targs)
            targ_var = targs_normed.requires_grad_(True)
            # find output
            output = model(*input_var)
            loss = loss_fn(output,targ_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(train_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
        return last_loss
    # end
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0
    if not peaks:
        maes = np.empty((EPOCHS,4))
    else:
        maes = np.ones((EPOCHS,))
    losses = np.array([],dtype=np.float32)
    thres = 1e-6
    # iter over epochs
    for epoch in range(EPOCHS):
        # train
        model.train(True)
        avg_loss = train_one_epoch(epoch_number,writer)

        running_vloss = 0.0
        running_vmae1 = 0.0
        running_vmae2 = 0.0
        running_vmae3 = 0.0
        running_vmae4 = 0.0
        # eval time
        model.eval()
        with torch.no_grad():
            for i,(vins,vtargs,_) in enumerate(valid_loader):
                vin_var = (vins[0].requires_grad_(True),
                        vins[1].requires_grad_(True),
                        vins[2],
                        vins[3])
                vtargs_normed = normalizer.norm(vtargs)
                vtargs_var = vtargs_normed.requires_grad_(True)
                vouts = model(*vin_var)
                vloss = loss_fn(vouts, vtargs_var)
                running_vloss += vloss
                denormed_outs = normalizer.denorm(vouts)
                denormed_ins = normalizer.denorm(vtargs_var)
                # MAE loss?
                if not peaks:
                    vmae1 = mae(denormed_outs,denormed_ins,0)
                    running_vmae1 += vmae1
                    vmae2 = mae(denormed_outs,denormed_ins,1)
                    running_vmae2 += vmae2
                    vmae3 = mae(denormed_outs,denormed_ins,2)
                    running_vmae3 += vmae3
                    vmae4 = mae(denormed_outs,denormed_ins,3)
                    running_vmae4 += vmae4
        avg_vloss = running_vloss / (i+1)
        if not peaks:
            avg_vmae1 = running_vmae1 / (i+1)
            avg_vmae2 = running_vmae2 / (i+1)
            avg_vmae3 = running_vmae3 / (i+1)
            avg_vmae4 = running_vmae4 / (i+1)
        print("Epoch"+str(epoch)+", validation loss:",avg_vloss)
        if not peaks:
            maes[epoch,:] = np.array([avg_vmae1,avg_vmae2,avg_vmae3,avg_vmae4])
        losses = np.append(losses,np.array([avg_vloss]))
        # step the optimizer
        if not adam:
            scheduler.step(avg_vloss)
        # etc
        writer.add_scalars('Training vs. Validation Loss',{ 'Training' : avg_loss, 'Validation' : avg_vloss }, epoch_number + 1)
        writer.flush()
        epoch_number += 1
        # break if smol lr
        print(scheduler.get_last_lr()[0])
        if not adam:
            if infin and (scheduler.get_last_lr()[0] <= thres):
                print("Ending due to small learning rate.")
                break

    # return the stuff.
    return maes[0:(epoch+1)],losses,model

def testmodel(inmodel,N=20,mode='valid',scale=4410,lognorm=False,fname='id_prop.npy'):
    root_dir = "C:/Users/bjama/Desktop/big_NN/big_NN/data"
    dataset = JSONData(root_dir,fname=fname)
    collate_fn = collate_pool
    train_loader, valid_loader, _ = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=1,
        train_ratio=0.6,
        num_workers=0,
        val_ratio=0.2,
        test_ratio=0.1,
        pin_memory=False,
        gen=None,
        train_size=None,
        val_size=None,
        test_size=None,
        return_test=True,)
    if mode == 'valid':
        loader = valid_loader
    elif mode == 'train':
        loader = train_loader
    else:
        print("Invalid mode.")
        return
    if lognorm:
        normalizer = logNormalizer(scale_fac=scale)
    else:
        normalizer = Normalizer(scale_fac=scale)
    
    output = np.array((N,2,8))

    inmodel.eval()
    with torch.no_grad():
        for i,(tins,targs,_) in enumerate(loader):
            if i == N:
                break
            tin_var = (tins[0].requires_grad_(True),
                        tins[1].requires_grad_(True),
                        tins[2],
                        tins[3])
            normed_outs = inmodel(*tin_var)
            denorm_outs = normalizer.denorm(normed_outs)
            output[i,0,:] = denorm_outs.detach().numpy()
            output[i,1,:] = targs.detach().numpy()
    return output

def dummymodel(atom_fea_len=64,n_conv=3,n_h=25,h_fea_len=128,fname='id_prop.npy'):
    root_dir = "C:/Users/bjama/Desktop/big_NN/big_NN/data"
    dataset = JSONData(root_dir,fname=fname)
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = eg.CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=atom_fea_len,
                                    n_conv=n_conv,
                                    n_h=n_h,
                                    h_fea_len=h_fea_len,output_len=8)
    return model
