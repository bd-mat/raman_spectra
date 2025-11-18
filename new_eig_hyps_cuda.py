import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from eigenfreq_dataset_loader import JSONData
from eigenfreq_dataset_loader import collate_pool
from eigenfreq_dataset_loader import get_train_val_test_loader


import eig_model as eg

def mae(output,target,ind):
    dif = torch.abs(target-output)[:,int(ind)]
    goods = torch.argwhere(target[:,int(ind)+4]!=0).flatten()
    return torch.mean(dif[goods])

def mae_int(output,target):
    dif = torch.abs(target-output)[:,4:]
    return torch.mean(dif)

def dif_npeaks(output,target):
    TOL = 0.1
    npeaks_targ = torch.sum((target[:,4:] != 0).to(torch.float),dim=1)
    npeaks_out = torch.sum((output[:,4:] >= TOL).to(torch.float),dim=1)
    return torch.mean(npeaks_out - npeaks_targ)

def wloss(k1,k2):
    def loss(output, target):
        diff = torch.abs(target-output)
        diff[:,:4] *= target[:,4:]*k1
        diff[:,4:] *= k2
        
        return torch.mean(diff)
    return loss

def wloss2(k):
    def loss(output, target):
        diff = torch.abs(target-output)
        diff[:,:4] *= target[:,4:]
        diff[:,4:] *= k
        
        return torch.mean(diff)
    return loss

def loss3(output, target):
    diff = torch.abs(target-output)
    diff[:,:4] *= target[:,4:]
    diff[:,:4] /= target[:,4:] + 0.0000001
    return torch.mean(diff[:,:4])

def loss4(k):
    def loss(output, target):
        order = output[:,:4].argsort(dim=1)
        output[:,:4] = torch.gather(output[:,:4], 1, order)
        output[:,4:] = torch.gather(output[:,4:], 1, order)
        diff = torch.abs(target-output)
        
        diff[:,:4] *= target[:,4:]
        diff[:,:4] /= (target[:,4:] + 0.00000001)
        
        diff[:,4:] *= k
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
    lr=0.4,
    EPOCHS=30,
    p_train = 0.7,
    p_val = 0.2,
    p_test = 0.1,
    seed=123,
    n_h = 1,
    n_conv = 3,
    h_fea_len = 64,
    atom_fea_len = 64,
    scale=4410,
    k=0.1,
    bsize=256,
    n_r=3,
    n_r_fea=64,
    n_r_h_fea=32,
    loss_fn = 'loss3',
    patience=10,
    patience_thres = 0.001,
    early_stopping = True,
    id_prop_fname='id_prop_byfreq_noneg.npy',
    do_cuda = False
):
    n_r_fea = h_fea_len
    n_r_h_fea = n_r_fea
    
    torch.manual_seed(seed)
    mom = 0.9
    wd = 0
    batch_size = bsize
    num_workers = 0
    scale_fac = 4410
    if do_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else: device = "cpu"
    
    #root_dir = "/home/yihao/mphys_bens/data"
    root_dir = "C:/Users/reise/Documents/Uni/Y4/MPhys_Project/big_NN/data"

    
    hyperparam_dict = {"n_h": n_h, "k": k, "lr": lr, "mom": mom,
                       "EPOCHS": EPOCHS, "p_train": p_train, "p_val": p_val, 
                       "p_test": p_test, "seed": seed, "n_conv": n_conv, 
                       "h_fea_len": h_fea_len, "atom_fea_len": atom_fea_len,
                       "loss_fn": loss_fn, "patience": patience,
                       "id_prop_fname":id_prop_fname, "n_r":n_r,
                       "n_r_h_fea":n_r_h_fea, "scale":scale,
                       "n_r_fea":n_r_fea, "n_r_h_fea":n_r_h_fea}

    dataset = JSONData(root_dir, id_prop_fname=id_prop_fname)

    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    if loss_fn =='loss3':
        loss_fn = loss3 
    
    elif loss_fn == 'loss4':
        loss_fn = loss4(k)
    elif loss_fn == 'wloss2':
        loss_fn = wloss2(k)
    else: print('No loss function that matches')
    

    model = eg.cr_cgcnn(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=atom_fea_len,
                                n_conv=n_conv,
                                n_h=n_h,
                                h_fea_len=h_fea_len,output_len=8,
                                n_r=n_r, cgcnn_len=n_r_fea, r_hidden_fea=n_r_h_fea)
    model.to(device)
    

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,"min",patience=6,threshold=0.0003)
    
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

    
    normalizer = Normalizer(scale_fac=scale_fac)
    # epoch training
    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.
        for i,(ins,targs,_) in enumerate(train_loader):
            # load inputs
            ins_2_dev = [x.to(device) for x in ins[2]] if isinstance(ins[2], list) else ins[2].to(device)
            ins_3_dev = [x.to(device) for x in ins[3]] if isinstance(ins[3], list) else ins[3].to(device)
            input_var = (ins[0].to(device).requires_grad_(True),
                        ins[1].to(device).requires_grad_(True),
                        ins_2_dev,
                        ins_3_dev)
            # normalize targets
            targs_normed = normalizer.norm(targs)
            targ_var = targs_normed.to(device).requires_grad_(True)
            # find output
            output = model(*input_var)
            loss = loss_fn(output,targ_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100 # loss per batch
                tb_x = epoch_index * len(train_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
        return last_loss
    # end
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    maes = np.empty(EPOCHS)
    IMAEs = np.empty(EPOCHS)
    corr_num_peaks = np.empty(EPOCHS)

    losses = np.array([],dtype=np.float32)
    
    # --- Early Stopping Variables ---
    epochs_no_improve = 0
    # --------------------------------
    
    # iter over epochs
    for epoch in range(EPOCHS):
        # train
        model.train(True)
        avg_loss = train_one_epoch(epoch_number,writer)

        running_vloss = 0.0
        running_vmae, running_vmae1, running_vmae2, running_vmae3, running_vmae4 = 0, 0, 0, 0, 0
        running_imae = 0.0
        running_peaks = 0.0
        nc1, nc2, nc3, nc4 = 0,0,0,0

        # eval time
        model.eval()
        with torch.no_grad():
            for i,(vins,vtargs,_) in enumerate(valid_loader):
                
                vins_2_dev = [x.to(device) for x in vins[2]] if isinstance(vins[2], list) else vins[2].to(device)
                vins_3_dev = [x.to(device) for x in vins[3]] if isinstance(vins[3], list) else vins[3].to(device)
                vin_var = (vins[0].to(device).requires_grad_(True),
                        vins[1].to(device).requires_grad_(True),
                        vins_2_dev,
                        vins_3_dev)
                vtargs_normed = normalizer.norm(vtargs)
                vtargs_var = vtargs_normed.to(device).requires_grad_(True)
                vouts = model(*vin_var)
                vloss = loss_fn(vouts, vtargs_var)
                running_vloss += vloss
                denormed_outs = normalizer.denorm(vouts).cpu()
                denormed_ins = normalizer.denorm(vtargs_var).cpu()
                # MAE loss?

                vmae1 = mae(denormed_outs,denormed_ins,0)
                if vmae1 == 0:
                    nc1 += 1
                running_vmae1 += vmae1
                vmae2 = mae(denormed_outs,denormed_ins,1)
                if vmae2 == 0:
                    nc2 += 1
                running_vmae2 += vmae2
                vmae3 = mae(denormed_outs,denormed_ins,2)
                if vmae3 == 0:
                    nc3 += 1
                running_vmae3 += vmae3
                vmae4 = mae(denormed_outs,denormed_ins,3)
                if vmae4 == 0:
                    nc3 += 1
                running_vmae4 += vmae4
                imae = mae_int(denormed_outs,denormed_ins)
                running_imae += imae
                npeaks = dif_npeaks(denormed_outs,denormed_ins)
                
                running_peaks += npeaks

        avg_vmae1 = running_vmae1 / (i-nc1+1)
        avg_vmae2 = running_vmae2 / (i-nc2+1)
        avg_vmae3 = running_vmae3 / (i-nc3+1)
        avg_vmae4 = running_vmae4 / (i-nc4+1)
        avg_imae = imae / (i+1)
        avg_peaks = running_peaks / (i+1)
        
        avg_vloss = (running_vloss / (i+1)).item()

        avg_vmae = np.mean([avg_vmae1, avg_vmae2, avg_vmae3, avg_vmae4])
        

        maes[epoch] = avg_vmae
        IMAEs[epoch] = avg_imae
        corr_num_peaks[epoch] = avg_peaks
        
        losses = np.append(losses,np.array([avg_vloss]))
        # step the optimizer

        scheduler.step(avg_vloss)
        # etc
        writer.add_scalars('Training vs. Validation Loss',{ 'Training' : avg_loss, 'Validation' : avg_vloss }, epoch_number + 1)
        writer.flush()
        epoch_number += 1

        # --- Early Stopping Check ---
        if early_stopping:
            if epoch_number>12:
            	if avg_vloss < np.min(losses[-patience:-1]):
            		epochs_no_improve = 0
            	else:
            		epochs_no_improve += 1
            		if epochs_no_improve == patience:
            			#print(f"Early stopping: {losses[-patience:]}")
            			break # Exit the training loop
        # ----------------------------
    
    torch.cuda.empty_cache()

    # return the stuff.
    return maes[0:(epoch+1)],losses, hyperparam_dict, model, IMAEs[0:(epoch+1)], corr_num_peaks[0:(epoch+1)]

def testmodel(inmodel,N=20,mode='valid',scale=4410,lognorm=False,id_prop_fname='id_prop.npy', do_cuda=True):
    #root_dir = "/home/yihao/mphys_bens/data"
    root_dir = "C:/Users/reise/Documents/Uni/Y4/MPhys_Project/big_NN/data"
    dataset = JSONData(root_dir, id_prop_fname=id_prop_fname)
    collate_fn = collate_pool
    pin_memory = False
    if do_cuda: pin_memory=True
    
    train_loader, valid_loader, _ = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=1,
        train_ratio=0.6,
        num_workers=4,
        val_ratio=0.2,
        test_ratio=0.1,
        pin_memory=pin_memory,
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
    
    output = np.empty((N,2,8))
    
    if do_cuda: 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inmodel.to(device)
    else: device = "cpu"

    inmodel.eval()
    with torch.no_grad():
        for i,(tins,targs,_) in enumerate(loader):
            if i == N:
                break
            
            tins_2_dev = [x.to(device) for x in tins[2]] if isinstance(tins[2], list) else tins[2].to(device)
            tins_3_dev = [x.to(device) for x in tins[3]] if isinstance(tins[3], list) else tins[3].to(device)
            tin_var = (tins[0].to(device).requires_grad_(True),
                        tins[1].to(device).requires_grad_(True),
                        tins_2_dev,
                        tins_3_dev)
            normed_outs = inmodel(*tin_var)
            denorm_outs = normed_outs #normalizer.denorm(normed_outs)
            output[i,0,:] = denorm_outs.cpu().detach().numpy()
            output[i,1,:] = targs.cpu().detach().numpy()
    
    torch.cuda.empty_cache()
    
    return output





