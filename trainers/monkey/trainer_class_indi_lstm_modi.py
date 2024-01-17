# Shirunwu Friday, December 15, 2023 @ 14:01:45 PM
import os
import yaml
import json
import torch
import argparse
import models
import numpy as np
from tqdm import tqdm
from datasets import MelDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

# path
os.chdir('/mnt/work/')

# evaluate val dataset
def evaluate_step(model, dataloader):
    loss = 0.
    kld_f = 0.
    kld_z = 0.
    sample_size = 0
    for mel, lenx, indi_mel, ctID, cID in dataloader:
        mel = mel.to(model.device)
        lenx = lenx.to(model.device)
        indi_mel = indi_mel.to(model.device)
        
        outputs = model(mel, indi_mel) 
        _loss, _kld_f, _kld_z = model.loss_fn(mel, 
                                            outputs['recon_x'], 
                                            outputs['f_mean'], 
                                            outputs['f_logvar'], 
                                            outputs['z_post_mean'], 
                                            outputs['z_post_logvar'], 
                                            outputs['z_prior_mean'], 
                                            outputs['z_prior_logvar'])
        
        
        batch_size = mel.shape[0]
        sample_size += batch_size
        
        loss += _loss.item() * batch_size
        kld_f += _kld_f.item() * batch_size
        kld_z += _kld_z.item() * batch_size
  
    return (loss / sample_size, kld_f / sample_size, kld_z / sample_size)



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
    

def log(logger, step=None, losses=None, fig=None, audio=None, sampling_rate=44100, tag="", model=None):
    if losses is not None:
        logger.add_scalar("Loss/meanloss", losses[0], step)
        logger.add_scalar("Loss/meanf", losses[1], step)
        logger.add_scalar("Loss/meanz", losses[2], step)
        
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
    used_key = [[
                #   'calltype_1',
                    'calltype_2',
                    'calltype_3',
                    'calltype_4',
                    'calltype_5',
                  'calltype_6',
                  'calltype_7',
                  'calltype_8',
                  'calltype_9',
                  'calltype_10'
                    ],
                        [
                        'twin_1_0',
                        'twin_1_1',
                        'twin_2_2',
                        'twin_2_3',
                        'twin_3_4',
                        'twin_3_5',
                        'twin_4_6',
                        'twin_4_7',
                        'twin_5_8',
                        'twin_5_9'
                        ]]
    trn_set = MelDataset(dataset_config, used_key = used_key, subset='train')
    val_set = MelDataset(dataset_config, used_key = used_key, subset='val')
    tst_set = MelDataset(dataset_config, used_key = used_key, subset='test')
    
    print('len', len(trn_set))
    
    batch_size = train_config["optimizer"]["batch_size"]
    
    # Dataloader
    trn_loader = DataLoader(
        trn_set, batch_size=batch_size, num_workers=4, shuffle=True,
        collate_fn=trn_set.collate_fn, pin_memory=True)
    
    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=4, shuffle=True,
        collate_fn=val_set.collate_fn, pin_memory=True)
    
    val_single_sampler = DataLoader(val_set, batch_size=1, shuffle=True)
    tst_single_sampler = DataLoader(tst_set, batch_size=1, shuffle=True)

    print('over loading data', '\n',
          'training size:', len(trn_set))
    
    # model loading 
    model_name = model_config['model_name']
    model_type = getattr(models, model_name) # model choose
    model = model_type(device = device, 
                       module_config = {'static_encoder': model_config['static_encoder']}, 
                       **model_config['DisentangledVAE1D']).to(device) # model config load
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
    total_epoch = train_config["step"]["total_epoch"]
    
    # Experiment name
    exp_name = '{}-{}'.format(model_name,experi_name)
    current_directory = os.path.dirname(os.path.abspath(__file__))
    exp_name =  current_directory + '/' + "output" + '/' + exp_name

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
        
    
    if train_config["load_model"]> 0:
        global_step = train_config["load_step"] + 1
        start_epoch = train_config["start_epoch"] + 1
    else:
        global_step = 0
        start_epoch = 0
        
    # start training---------------------------------------------------------------
    for epoch in range(start_epoch, total_epoch):
        losses = []
        kld_fs = []
        kld_zs = []
        
        for mel, lenx, indi_mel, ctID, cID in tqdm(trn_loader):
            mel = mel.to(device)
            lenx = lenx.to(device)
            indi_mel = indi_mel.to(device)

            # gradient zero
            model.zero_grad(set_to_none=True)
            
            outputs = model(mel, indi_mel) 
            loss, kld_f, kld_z = model.loss_fn(original_seq=mel, 
                                            recon_seq=outputs['recon_x'], 
                                            f_mean=outputs['f_mean'], 
                                            f_logvar=outputs['f_logvar'], 
                                            z_post_mean=outputs['z_post_mean'], 
                                            z_post_logvar=outputs['z_post_logvar'], 
                                            z_prior_mean=outputs['z_prior_mean'], 
                                            z_prior_logvar=outputs['z_prior_logvar'])
            
            loss.backward()
            optim.step()
            
            losses.append(loss.item())
            kld_fs.append(kld_f.item())
            kld_zs.append(kld_z.item())

            if global_step % log_step == 0:
                # evaluate epoch          
                meanloss = np.mean(losses)
                meanf = np.mean(kld_fs)
                meanz = np.mean(kld_zs)
                
                losses = [meanloss, meanf, meanz]
                message1 = "\n Epoch {}/{}, Step {}/{}, ".format(epoch+1, total_epoch, global_step, total_step)
                message2 = "\n Average Loss: {:.4f} KL of f : {:.4f} KL of z : {:.4f}".format(*losses)
                with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                    f.write(message1 + message2 + "\n")
                print(message1 + message2)
                log(train_logger, global_step, losses=losses, model=model_name)


            if global_step % val_step == 0:
                model.eval()
                val_loss, val_kld_f, val_kld_z = evaluate_step(model, val_loader)
                log(val_logger, step=global_step, model=model_name,
                    losses=[val_loss, val_kld_f, val_kld_z])
                message = "\n Average Loss: {:.4f} KL of f : {:.4f} KL of z : {:.4f}"\
                    .format(val_loss, val_kld_f, val_kld_z)
                with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                    f.write(message + "\n")
                print(message)
                val_losses.append([val_loss, val_kld_f, val_kld_z])
                
                # reconstruction
                val_mel, val_lenx, val_indi_mel, val_ctID, val_cID = next(iter(val_single_sampler))
                val_mel = val_mel.to(device)
                val_lenx = val_lenx.to(device)
                val_indi_mel = val_indi_mel.to(device)
                
                with torch.no_grad():
                    outputs = model(val_mel, val_indi_mel) 
                    rec_mel = outputs['recon_x']
                    rec_mel_fig = plot_mel(rec_mel)
                    gr_mel_fig = plot_mel(val_mel)

                    log(val_logger, step=global_step, fig=gr_mel_fig,
                        tag="Val/step-{}-{}_mel_gt".format(global_step, val_ctID.detach().cpu().numpy()))
                    log(val_logger, step=global_step, fig=rec_mel_fig,
                        tag="Val/step-{}-{}_mel".format(global_step, val_ctID.detach().cpu().numpy()))
                        
                model.train()
                
            if global_step % save_step == 0:
                torch.save(
                    {"model": model.state_dict(), "optimizer": optim.state_dict()},
                    os.path.join(exp_name, train_config["path"]["save_path"], "{}.pth.tar".format(global_step)))
            
            global_step += 1
            if global_step > total_step:
                avg_val_losses = np.array(val_losses).mean(axis=0).tolist()
                print("Overall: Loss: {:.3f}, KL of f: {:.3f}, KL of z: {:.3f}".format(*avg_val_losses))
                quit()

        

if __name__ == "__main__":
    # f input is suffled
    
    
    # only for caller 
    dataset_pathname = "dataset4.yaml"
    model_pathname = "model_lstm3.yaml"
    train_pathname = "train_lstm12.yaml" 
    
    # config path
    data_config_path  = "./Animal/configs/monkey" + "/" + dataset_pathname
    model_config_path = "./Animal/configs/monkey" + "/" + model_pathname
    train_config_path = "./Animal/configs/monkey" + "/" + train_pathname
     
    # read Config
    dataset_config = yaml.load(open(data_config_path, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config_path, "r"), Loader=yaml.FullLoader)
    configs = (dataset_config, model_config, train_config)
    file_config = (dataset_pathname, model_pathname, train_pathname)
    
    # run
    experi_name = 'LSTM4'
    main(configs, file_config, experi_name)
    

    