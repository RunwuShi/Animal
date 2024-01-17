import os
import sys
import json
from argparse import ArgumentParser
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
# personal files
os.chdir('/mnt/work/')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import models
from datasets import MelDataset, vctkDataset, greatbirdDataset


# evaluate val dataset
def evaluate_step(model, dataloader):
    nll = 0.
    con_kl = 0.
    indi_kl = 0.
    sample_size = 0
    for mel, lenx, indi_mel, cID, cID_type in dataloader:
        mel = mel.to(model.device)
        indi_mel = indi_mel.to(model.device)
        lenx = lenx.to(model.device)
        
        outputs = model(mel, lenx, indi_mel)
        _nll, _indi_kl, _con_kl = model.loss_fn(outputs, mel, lenx)
        
        batch_size = mel.shape[0]
        sample_size += batch_size
        
        nll += _nll.item() * batch_size
        con_kl += _con_kl.item() * batch_size
        indi_kl += _indi_kl.item() * batch_size
  
    return (nll / sample_size, indi_kl / sample_size, con_kl / sample_size)

def plot_mel(mel_data):
    mel_data = mel_data.detach().cpu().numpy()
    print('mel_data',mel_data.shape)
    fig, axes = plt.subplots(len(mel_data), 1, squeeze=False)
    
    for i in range(len(mel_data)):
        mel = mel_data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")
        
    return fig
    

def log(logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22100, tag="", model=None):
    if losses is not None:
        logger.add_scalar("Loss/NLL", losses[0], step)
        logger.add_scalar("Loss/CON-KL", losses[1], step)
        logger.add_scalar("Loss/INDI-KL", losses[2], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(tag, audio / max(abs(audio)), sample_rate=sampling_rate)

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def main(configs, file_config, experi_name):
    # device 1 
    torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_config, model_config, train_config = configs
    
    # data loading
    print('start loading data')
    trn_set = greatbirdDataset(dataset_config, subset='train') # 'train'
    val_set = greatbirdDataset(dataset_config, subset='val')
    # tst_set = greatbirdDataset(dataset_config, subset='test')
    
    print('len', len(trn_set))
    
    batch_size = train_config["optimizer"]["batch_size"]
    
    # Dataloader
    trn_loader = DataLoader(
        trn_set, batch_size=batch_size, num_workers=6, shuffle=True,
        collate_fn=trn_set.collate_fn, pin_memory=True)
    
    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=4, shuffle=True,
        collate_fn=val_set.collate_fn, pin_memory=True)
    
    val_single_sampler = DataLoader(val_set, batch_size=1, shuffle=True)
    

    print('over loading data', '\n',
          'training size:', len(trn_set))
    
    # model loading 
    model_name = model_config['model_name']
    model_type = getattr(models, model_name) # model choose
    model = model_type(model_config,device).to(device) # model config load
    num_param = get_param_num(model)
    
    print('Model name:', model_name, 'Number of Parameters:', num_param)
    
    # training para loading
    # set optimizers
    learning_rate = train_config["optimizer"]["learning_rate"]
    betas = train_config["optimizer"]["betas"]
    eps = train_config["optimizer"]["eps"]
    optim = torch.optim.AdamW(model.parameters(), learning_rate, betas=betas, eps=eps)
    
    # training hyper para
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    val_step = train_config["step"]["val_step"]
    con_gamma = train_config["optimizer"]["con_gamma"]
    indi_gamma = train_config["optimizer"]["indi_gamma"]
    con_mi = train_config["optimizer"]["con_mi"]
    indi_mi = train_config["optimizer"]["indi_mi"]
    stop_step = train_config["step"]["mi_stop"]
    
    # Experiment name
    exp_name = '{}-{}-c_{}_{}-i_{}_{}'.format(
        model_name, experi_name.split('/')[-1], con_gamma, con_mi, indi_gamma, indi_mi)
    # current_directory = os.path.dirname(os.path.abspath(__file__))
    exp_name =   "./Animal/output" + '/'+ experi_name.split('/')[0] + '/' + exp_name

    # Load model checkpoint
    if train_config["load_model"]:
        save_path = os.path.join(exp_name, train_config["path"]["save_path"], 
                                 "{}.pth.tar".format(train_config["load_step"]))
        ckpt = torch.load(save_path)
        model.load_state_dict(ckpt["model"])
    
    # Init logger
    for p in train_config["path"].values():
        os.makedirs(os.path.join(exp_name, p), exist_ok=True)
    train_log_path = os.path.join(exp_name, train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(exp_name, train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)
    val_losses = []
    
    # save log file config 
    total_config = configs
    file_config = file_config
    with open(exp_name+'/'+'total_config.json', 'w') as f:
        json.dump(total_config, f, indent = 4)
    with open(exp_name+'/'+'file_config.json', 'w') as f:
        json.dump(file_config, f, indent = 4)
        
    
    if train_config["load_model"] and train_config["load_step"] > 0:
        global_step = train_config["load_step"] + 1
    else:
        global_step = 0
        
    # start training---------------------------------------------------------------
    while True:
        for mel, lenx, indi_mel, cID, cID_type in tqdm(trn_loader):
            mel = mel.to(device)
            lenx = lenx.to(device)
            indi_mel = indi_mel.to(device)

            # gradient zero
            model.zero_grad(set_to_none=True)
            
            outputs = model(mel, lenx, indi_mel) 
            nll, indi_kl, con_kl = model.loss_fn(outputs, mel, lenx)
            
            indi_c = np.clip((indi_mi / stop_step) * global_step, 0, indi_mi)
            con_c = np.clip((con_mi / stop_step) * global_step, 0, con_mi)
            
            # total loss function
            loss = (nll + con_gamma * (con_kl - con_c).abs() +
                    indi_gamma * (indi_kl - indi_c).abs())
            
            loss.backward()
            optim.step()
            
            # print('run')
            # evaluate
            if global_step > 0 and global_step % log_step == 0:
                losses = [nll.item(), con_kl.item(), indi_kl.item()]
                message1 = "Step {}/{}, ".format(global_step, total_step)
                message2 = "NLL: {:.3f}, con-kl: {:.3f}, indi-kl: {:.3f},".format(*losses)
                with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                    f.write(message1 + message2 + "\n")
                print(message1 + message2)
                log(train_logger, global_step, losses=losses, model=model_name)
    
            if global_step > 0 and global_step % val_step == 0:
                model.eval()
                val_nll, val_indi_kl, val_con_kl = evaluate_step(model, val_loader)
                log(val_logger, step=global_step, model=model_name,
                    losses=[val_nll, val_con_kl, val_indi_kl])
                message = "Val-NLL: {:.3f}, val-con-kl: {:.3f}, val-indi-kl: {:.3f}"\
                    .format(val_nll, val_con_kl, val_indi_kl)
                with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                    f.write(message + "\n")
                print(message)
                val_losses.append([val_nll, val_indi_kl, val_con_kl])
                
                # reconstruction
                val_mel, val_lenx, val_indi_mel, val_cID, val_cID_type= next(iter(val_single_sampler))
                val_mel = val_mel.to(device)
                val_lenx = val_lenx.to(device)
                val_indi_mel = val_indi_mel.to(device)
                
                # tst_mel, tst_lenx, tst_indi_mel, tst_ctID, tst_cID = next(iter(tst_single_sampler))
                # tst_mel = tst_mel.to(device)
                # tst_lenx = tst_lenx.to(device)
                # tst_indi_mel = tst_indi_mel.to(device)

                with torch.no_grad():
                    outputs = model(val_mel, val_lenx, val_indi_mel)
                    
                    rec_mel = outputs['x_rec']
                    rec_mel_fig = plot_mel(rec_mel)
                    gr_mel_fig = plot_mel(val_mel)

                    log(val_logger, step=global_step, fig=gr_mel_fig,
                        tag="Val/step-{}-{}_mel_gt".format(global_step, val_cID_type))
                    log(val_logger, step=global_step, fig=rec_mel_fig,
                        tag="Val/step-{}-{}_mel".format(global_step, val_cID_type))
                     
                model.train()
                
            if global_step > 0 and global_step % save_step == 0:
                torch.save(
                    {"model": model.state_dict(), "optimizer": optim.state_dict()},
                    os.path.join(exp_name, train_config["path"]["save_path"], "{}.pth.tar".format(global_step)))
            
            global_step += 1
            if global_step > total_step:
                avg_val_losses = np.array(val_losses).mean(axis=0).tolist()
                print("Overall: Val-NLL: {:.3f}, val-indi-kl: {:.3f}, val-con-kl: {:.3f}".format(*avg_val_losses))
                quit()
            

if __name__ == "__main__":

    # config file name
    dataset_pathname = "dataset_greatbird.yaml"
    model_pathname = "model_greatbird.yaml"
    train_pathname = "train_greatbird6.yaml" 
    
    # config path
    data_config_path  = "./Animal/configs/greatbird/" + "/" + dataset_pathname
    model_config_path = "./Animal/configs/greatbird/" + "/" + model_pathname
    train_config_path = "./Animal/configs/greatbird/" + "/" + train_pathname
     
    # read Config
    dataset_config = yaml.load(open(data_config_path, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config_path, "r"), Loader=yaml.FullLoader)
    configs = (dataset_config, model_config, train_config)
    file_config = (dataset_pathname, model_pathname, train_pathname)
    
    # run
    experi_name = 'greatbird/greatbird_6'
    main(configs, file_config, experi_name)
    

    