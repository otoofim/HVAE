import torchvision.models as models
from torchvision import transforms
from dataloader_cityscapes import *
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torchvision
import wandb
import os
import warnings
from statistics import mean
from ProUNet import *
from torch.distributions import Normal, Independent, kl, MultivariateNormal
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def train(**kwargs):


    #kwargs["model_add"] = "./checkpoints/{}_{}/best.pth".format(kwargs["project_name"], kwargs["run_name"])
    
    hyperparameter_defaults = {
        "run": kwargs["run_name"],
        "hyper_params": kwargs,
    }

    base_add = os.getcwd()


    if kwargs['continue_tra']:
        wandb.init(config = hyperparameter_defaults, project = kwargs["project_name"], entity = 'moh1371',
                    name = hyperparameter_defaults['run'], resume = "must", id = kwargs["wandb_id"])
        print("wandb resumed...")
    else:
        wandb.init(config = hyperparameter_defaults, project = kwargs["project_name"], entity = 'moh1371',
                    name = hyperparameter_defaults['run'], resume = "allow")


    val_every = 2
    img_w = kwargs["input_img_dim"][0]
    img_h = kwargs["input_img_dim"][1]


    preprocess_in = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        transforms.Resize(kwargs["input_img_dim"])
    ])

    preprocess_ou = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(kwargs["input_img_dim"])
    ])

    tr_loader = CityscapesLoader(dataset_path = kwargs["data_path"], transform_in = preprocess_in, transform_ou = preprocess_ou, mode = 'train')
    train_loader = DataLoader(dataset = tr_loader, batch_size = kwargs["batch_size"], shuffle = True, drop_last = True)
    kwargs["no_classes"] = tr_loader.get_num_classes()

    val_loader = CityscapesLoader(dataset_path = kwargs["data_path"], transform_in = preprocess_in, transform_ou = preprocess_ou, mode = 'val')
    val_loader = DataLoader(dataset = val_loader, batch_size = kwargs["batch_size"], shuffle = True, drop_last = True)




    if kwargs['device'] == "cpu":
        device = torch.device("cpu")
        print("Running on the CPU")
    elif kwargs['device'] == "gpu":
        device = torch.device(kwargs['device_name'])
        print("Running on the GPU")



    model = ProUNet(num_classes = kwargs["no_classes"], LatentVarSize = kwargs["latent_dim"], beta = kwargs["beta"], training = True, num_samples = kwargs["num_samples"])
    
    if kwargs["continue_tra"]:
        model.load_state_dict(torch.load(kwargs["model_add"])['model_state_dict'])
        print("model state dict loaded...")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr =  kwargs["learning_rate"], weight_decay = kwargs["momentum"])
    criterion = nn.BCEWithLogitsLoss(size_average = False, reduce = False, reduction = None)


    if kwargs["continue_tra"]:
        optimizer.load_state_dict(torch.load(kwargs["model_add"])['optimizer_state_dict'])
        print("optimizer state dict loaded...")


    tr_loss = {"loss":0., "kl":0., "rec":0.}
    val_loss = {"loss":0., "kl":0., "rec":0.}
    best_val = 1e10
    wandb.watch(model)

    start_epoch = 0
    end_epoch = kwargs["epochs"]

    if kwargs["continue_tra"]:
        start_epoch = torch.load(kwargs["model_add"])['epoch'] + 1
        end_epoch = torch.load(kwargs["model_add"])['epoch'] + 1 + int(wandb.config.hyper_params["epochs"])


    with tqdm(range(start_epoch, end_epoch), unit="epoch", leave = True, position = 0) as epobar:
        for epoch in epobar:
                
                epobar.set_description("Epoch {}".format(epoch + 1))
                epobar.set_postfix(ordered_dict = {'tr_loss':tr_loss["loss"], 'val_loss':val_loss["loss"]})
                tr_loss = {"loss":0., "kl":0., "rec":0.}

                
                out = None
                images = None
                labels = None

                with tqdm(train_loader, unit="batch", leave = False) as batbar:
                    for i, batch in enumerate(batbar):
                        
                        batbar.set_description("Batch {}".format(i + 1))
                        optimizer.zero_grad()
                        model.train()
                        
                        #forward pass
                        seg, priorDists, posteriorDists = model(batch['image'].to(device), batch['seg'].to(device))
                        loss, kl_mean, kl_losses, rec_loss = model.elbo_loss(batch['label'].to(device), seg, priorDists, posteriorDists)
                        loss.backward()
                        optimizer.step()


                        tr_loss["loss"] += loss.item()
                        tr_loss["kl"] += kl_mean.item()
                        tr_loss["rec"] += rec_loss.item()
                        
                        for layer in kl_losses.items():
                            if "kl_layer{}".format(layer[0]) in tr_loss.keys():
                                tr_loss["kl_layer{}".format(layer[0])] += torch.mean(layer[1])
                            else:
                                tr_loss["kl_layer{}".format(layer[0])] = 0.
                                tr_loss["kl_layer{}".format(layer[0])] += torch.mean(layer[1])

                        images = batch['image'][-5:]
                        labels = batch['seg'][-5:]

                org_img = {'input': wandb.Image(images),
                "ground truth": wandb.Image(labels),
                "prediction": wandb.Image(tr_loader.prMask_to_color(seg[-5:].detach().cpu()))
                 }

                wandb.log(org_img)

                for key in tr_loss.keys():
                    tr_loss[key] /= len(train_loader)
                    wandb.log({key: tr_loss[key], "epoch": epoch + 1})


                if ((epoch+1) % val_every == 0):
                    with tqdm(val_loader, unit="batch", leave = False) as valbar:
                        with torch.no_grad():
                            val_loss = {"loss":0., "kl":0., "rec":0.}
                            
                            for i, batch in enumerate(valbar):
                                                                             
                                valbar.set_description("Val_batch {}".format(i + 1))
                                model.eval()
                                optimizer.zero_grad()
                                loss_sum = 0.
                                kl_sum = 0.
                                rec_sum = 0.
                                images = batch['image'][-5:]
                                labels = batch['seg'][-5 :]
                                                                             
                             
                                samples, priors, posteriorDists = model.evaluation(batch['image'].to(device), batch['seg'].to(device))
                                
                                for sample in samples:
                                                                             
                                    loss, kl_mean, kl_losses, rec_loss = model.elbo_loss(batch['label'].to(device), sample, priors, posteriorDists)
                                    
                                    loss_sum += loss
                                    kl_sum += kl_mean
                                    rec_sum += rec_loss

                                val_loss["loss"] += (loss_sum/kwargs["num_samples"]).item()
                                val_loss["kl"] += (kl_sum/kwargs["num_samples"]).item()
                                val_loss["rec"] += (rec_sum/kwargs["num_samples"]).item()

                        val_loss["loss"] /= len(val_loader)
                        val_loss["kl"] /= len(val_loader)
                        val_loss["rec"] /= len(val_loader)
                                                                                         
                        wandb.log({"loss_val": val_loss["loss"],
                                   "kl_val": val_loss["kl"],
                                   "rec_val": val_loss["rec"],
                                   "epoch": epoch + 1,
                                   "input_val": wandb.Image(images),
                                   "ground truth val": wandb.Image(labels),
                                   "prediction val": wandb.Image(tr_loader.prMask_to_color(torch.mean(samples,0).cpu().detach())[-5:])
                                  })



                        if val_loss["loss"] < best_val:

                            newpath = os.path.join(base_add, "checkpoints", hyperparameter_defaults['run'])

                            if not os.path.exists(os.path.join(base_add, "checkpoints")):
                                os.makedirs(os.path.join(base_add, "checkpoints"))

                            if not os.path.exists(newpath):
                                os.makedirs(newpath)

                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'tr_loss': tr_loss,
                                'val_loss': val_loss,
                                'hyper_params': kwargs,
                                }, os.path.join(newpath, "best.pth"))
