import os
import pickle
import pprint
import random
from glob import glob
from os.path import exists, join

import numpy as np
import torch
import sklearn, sklearn.metrics
import sklearn.model_selection
from sklearn.metrics import roc_auc_score, accuracy_score
import torchxrayvision as xrv

from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)
#from tqdm.auto import tqdm




def train(model, dataset, cfg):
    print("Our config:")
    pprint.pprint(cfg)
        
    dataset_name = cfg.dataset + "-" + cfg.model + "-" + cfg.name
    
    device = 'cuda' if cfg.cuda else 'cpu'
    if not torch.cuda.is_available() and cfg.cuda:
        device = 'cpu'
        print("WARNING: cuda was requested but is not available, using cpu instead.")

    print(f'Using device: {device}')

    print(cfg.output_dir)

    if not exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    
    # Setting the seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.cuda:
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Dataset    
    gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=cfg.seed)
    train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
    valid_dataset = xrv.datasets.SubsetDataset(dataset, test_inds)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=cfg.shuffle,
                                               num_workers=cfg.threads, 
                                               pin_memory=cfg.cuda)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=cfg.shuffle,
                                               num_workers=cfg.threads, 
                                               pin_memory=cfg.cuda)
    #print(model)

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5, amsgrad=True)
    print(optim)

    criterion = torch.nn.BCEWithLogitsLoss()

    # Checkpointing
    start_epoch = 0
    best_metric = 0.
    weights_for_best_validauc = None
    auc_test = None
    metrics = []
    weights_files = glob(join(cfg.output_dir, f'{dataset_name}-e*.pt'))  # Find all weights files
    if len(weights_files):
        # Find most recent epoch
        epochs = np.array(
            [int(w[len(join(cfg.output_dir, f'{dataset_name}-e')):-len('.pt')].split('-')[0]) for w in weights_files])
        start_epoch = epochs.max()
        weights_file = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()][0]
        
        with open(weights_file, 'rb') as f:
            state = torch.load(f, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            if "optimizer_state_dict" in state:
                optim.load_state_dict(state["optimizer_state_dict"])
                if device == "cuda":
                    for st in optim.state.values():
                        for k, v in st.items():
                            if isinstance(v, torch.Tensor):
                                st[k] = v.cuda()

        with open(join(cfg.output_dir, f'{dataset_name}-metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)

        best_metric = metrics[-1]['best_metric']
        weights_for_best_validauc = model.state_dict()

        print("Resuming training at epoch {0}.".format(start_epoch))
        print("Weights loaded: {0}".format(weights_file))

    if cfg.multicuda and (torch.cuda.device_count() > 1):
        print("Will try to use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)
    
    
    for epoch in range(start_epoch, cfg.num_epochs):

        avg_loss = train_epoch(cfg=cfg,
                               epoch=epoch,
                               model=model,
                               device=device,
                               optimizer=optim,
                               train_loader=train_loader,
                               criterion=criterion,
                               limit=cfg.limit)
        
        auc_valid = valid_test_epoch(name='Valid',
                                     epoch=epoch,
                                     model=model,
                                     device=device,
                                     data_loader=valid_loader,
                                     criterion=criterion,
                               limit=cfg.limit)[0]

        if np.mean(auc_valid) > best_metric:
            best_metric = np.mean(auc_valid)
            weights_for_best_validauc = model.state_dict()
            torch.save(model, join(cfg.output_dir, f'{dataset_name}-best.pt'))
            # only compute when we need to

        stat = {
            "epoch": epoch + 1,
            "trainloss": avg_loss,
            "validauc": auc_valid,
            'best_metric': best_metric
        }

        metrics.append(stat)

        with open(join(cfg.output_dir, f'{dataset_name}-metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)

        model_tosave = model
        if isinstance(model, torch.nn.DataParallel):
            model_tosave = model.module
        torch.save({'epoch': epoch,
                    'model_state_dict': model_tosave.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    }, join(cfg.output_dir, f'{dataset_name}-e{epoch + 1}.pt'))
        torch.save(model_tosave, join(cfg.output_dir, f'{dataset_name}-model{epoch + 1}.pt'))

    return metrics, best_metric, weights_for_best_validauc





def train_epoch(cfg, epoch, model, device, train_loader, optimizer, criterion, limit=None):
    model.train()
    
    avg_loss = []
    t = tqdm(train_loader)
    for batch_idx, samples in enumerate(t):
        
        if limit and (batch_idx > limit):
            print("breaking out")
            break
            
        optimizer.zero_grad()
        
        images = samples["img"].float().to(device)
        targets = samples["lab"].to(device)

        outputs = model(images)

        loss = (outputs["out"] - images)**2
        loss = loss.sum()
        
        if cfg.elastic:
            loss = loss + torch.abs(outputs["out"] - images).sum()
                
        loss = loss.sum()
        
        loss.backward()

        avg_loss.append(loss.detach().cpu().numpy())
        t.set_description(f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')

        optimizer.step()

    return np.mean(avg_loss)

def valid_test_epoch(name, epoch, model, device, data_loader, criterion, limit=None):
    model.eval()

    avg_loss = []
    task_outputs={}
    task_targets={}
    for task in range(data_loader.dataset[0]["lab"].shape[0]):
        task_outputs[task] = []
        task_targets[task] = []
        
    with torch.no_grad():
        t = tqdm(data_loader)
        for batch_idx, samples in enumerate(t):

            if limit and (batch_idx > limit):
                print("breaking out")
                break
            
            images = samples["img"].to(device)
            #targets = samples["lab"].to(device)

            outputs = model(images)

            #loss = torch.zeros(1).to(device).float()
            loss = (outputs["out"] - images)**2
            loss = loss.sum()
            
            avg_loss.append(loss.detach().cpu().numpy())
            t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')

    print(f'Epoch {epoch + 1} - {name}')

    return -np.mean(avg_loss), 0, task_outputs, task_targets
