import numpy as np
import torch
import time
import os
import glob

from classes import WavesDataset, MLP, LSTM_MLP, LSTM_CNN

def training(name):
    # name = 'mlp' # 'mlp', 'lstm_mlp', 'lstm_cnn'
    add = '' # saving purposes
    piml = False # whether to consider physical loss
    time_enc = True # whether to consider time encoding
    reset = True # whether to reset weights and log file
    adv_log = False # whether to save advanced log now
    mask = '' # mask name or ''
    lead_times = [0,8,16,24]# if name == 'lstm_cnn' else [0,12,24] # forecast lead times in hours

    timeenc_dim = 2 if time_enc else 0
    ds = WavesDataset('train', name, piml, timeenc_dim, mask, lead_times)

    if name == 'mlp':
        model = MLP(num_layers=4, width=16, activ_fun=torch.nn.LeakyReLU, input_dim=1+2+timeenc_dim+int(3*3*(1+12/6)), output_dim=3).to('cuda')

    elif name == 'lstm_mlp':
        model = LSTM_MLP(lstm_input_dim=2+3, lstm_hidden_dim=8, lstm_num_layers=1, lstm_output_dim=8,
                         mlp_add_input_dim=1+2+timeenc_dim, mlp_num_layers=4, mlp_width=32, mlp_activ_fun=torch.nn.LeakyReLU,
                         output_dim=3, device='cuda')

    else: #name == 'lstm_cnn'
        model = LSTM_CNN(lstm_input_dim=2+3, lstm_hidden_dim=8, lstm_num_layers=1, lstm_output_dim=8,
                         cnn_add_input_dim=1+timeenc_dim, cnn_activ_fun=torch.nn.LeakyReLU,
                         output_dim=3, device='cuda')

    val_frac   = 0.15
    train_size = int(ds.__len__() * (1-val_frac))
    val_size  = ds.__len__() - train_size
    batch_size = 2**21
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    name = name + '_' + add if add else name
    name = name + '_piml' if piml else name
    name = name + '_timeenc' if time_enc else name
    name = name + '_' + mask if mask else name

    os.makedirs(f'weights/{name}', exist_ok=True)
    weights_dir = glob.glob(f'weights/{name}/{name}_epoch_*.pth')
    if weights_dir and (not reset):
        latest_weights = max(weights_dir, key=lambda f: int(f.split('_epoch_')[-1].split('.pth')[0]))
        print(f'Loading weights: {latest_weights}')
        weights = torch.load(latest_weights)
        model.load_state_dict(weights['model_state_dict'])
        optimizer.load_state_dict(weights['optimizer_state_dict'])
        start_epoch = weights['epoch'] + 1  # resume from next epoch
        print(f'Resuming from {start_epoch}th epoch')
    else:
        for weight in weights_dir:
            os.remove(weight)
        start_epoch = 0
        print(f'No weights found or model reset. Starting from epoch {start_epoch}')

    end_epoch = int(1e5)
    pat_count = 0
    best_loss = float('inf')

    log_file = f"logs/log-{name}.npz"
    if os.path.exists(log_file) and not reset:
        data = np.load(log_file)
        log_data = {
            "epoch": list(data["epoch"]),
            "elapsed_time": list(data["elapsed_time"]),
            "train_loss": list(data["train_loss"]),
            "val_loss": list(data["val_loss"]),
        }
    else:
        log_data = {
            "epoch": [],
            "elapsed_time": [],
            "train_loss": [],
            "val_loss": []
        }
    os.makedirs("logs/advs/", exist_ok=True)

    for epoch in range(start_epoch, end_epoch+adv_log):
        train_loss = torch.zeros([4,1], device='cuda')
        val_loss = torch.zeros([4,1], device='cuda')
        # Train
        if not adv_log:
            tic = time.time()
            model.train()
            ds.shuffle_dt() #shuffle lead times each epoch
            if mask:
                ds.enable_mask()
            for jdx in range(len(lead_times)):
                perm = torch.randperm(train_size-lead_times[-1]*ds.inputs['ratio'], device='cuda') #shuffle training data each lead time package
                for i in range(0, train_size-lead_times[-1]*ds.inputs['ratio'], min(batch_size,train_size-lead_times[-1]*ds.inputs['ratio'])):
                    idx = perm[i:i + min(batch_size,train_size-lead_times[-1]*ds.inputs['ratio'])]
                    optimizer.zero_grad()
                    loss, eTm, eHscos, eHssin = model.pi_loss(ds.inputs, ds.targets, idx, jdx) if piml else model.loss(ds.inputs, ds.targets, idx, jdx)
                    loss.backward()
                    optimizer.step()
                    train_loss = torch.cat([train_loss,torch.cat([loss.detach().reshape(1,1),   eTm.detach().reshape(1,1),
                                                                  eHscos.detach().reshape(1,1), eHssin.detach().reshape(1,1)])], dim=1)
            toc = time.time()
        # Validation
        model.eval()
        if mask:
            ds.disable_mask()
        with torch.no_grad():
            ds.sort_dt()  # sort lead times for validation
            for jdx in range(len(lead_times)):
                for i in range(train_size, train_size+val_size-lead_times[-1]*ds.inputs['ratio'], min(batch_size,train_size+val_size-lead_times[-1]*ds.inputs['ratio'])):
                    idx = torch.arange(i,min(i+batch_size,train_size+val_size-lead_times[-1]*ds.inputs['ratio']), device='cuda')
                    loss, eTm, eHscos, eHssin = model.loss(ds.inputs, ds.targets, idx, jdx)
                    val_loss = torch.cat([val_loss,torch.cat([loss.detach().reshape(1,1), eTm.detach().reshape(1,1),
                                                              eHscos.detach().reshape(1,1), eHssin.detach().reshape(1,1)])], dim=1)
            if val_loss[0].mean() < best_loss:
                best_loss = val_loss[0].mean()
                pat_count = 0
            else:
                pat_count += 1
                if pat_count >= np.clip(0.1*epoch,10**2,10**3):
                    adv_log = True
            if adv_log or epoch == end_epoch-1:
                ds.ts_dt() # complete lead times for adv log
                for jdx in range(max(lead_times)+1):
                    adv_log_data = -1 * torch.ones([1, 9], device='cuda')
                    for i in range(train_size, train_size + val_size - lead_times[-1] * ds.inputs['ratio'], min(batch_size,train_size+val_size-lead_times[-1]*ds.inputs['ratio'])):
                        idx = torch.arange(i, min(i+batch_size,train_size+val_size-lead_times[-1]*ds.inputs['ratio']),
                                           device='cuda')
                        adv_log_data = torch.cat([adv_log_data, torch.cat([ds.targets['t'][idx].repeat_interleave(168//ds.inputs['ratio']).reshape(-1,1),
                                                 model.get_adv_log(ds.inputs, ds.targets, idx, jdx)],
                                                 dim=1)],dim=0)
                    np.savez_compressed(f"logs/advs/adv_log-{name}-{ds.inputs['dt'][0,:][jdx]}h.npz",
                                        adv_log_data[1:, :].detach().cpu().numpy())
                if adv_log:
                    return

        print(f'Epoch {epoch:4d}, Train loss: {train_loss[0].mean():6.4f}, '
              f'Val loss: {val_loss[0].mean():6.4f}, Elapsed time: {toc - tic:6.3f}s')

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    f'weights/{name}/{name}_epoch_{epoch}.pth')

        log_data["epoch"].append(epoch)
        log_data["elapsed_time"].append(toc - tic)
        log_data["train_loss"].append(train_loss.mean(axis=1).detach().cpu().numpy()[:4])
        log_data["val_loss"].append(val_loss.mean(axis=1).detach().cpu().numpy()[:4])
        np.savez_compressed(
            log_file,
            epoch=np.array(log_data["epoch"]),
            elapsed_time=np.array(log_data["elapsed_time"]),
            train_loss=np.array(log_data["train_loss"]),
            val_loss=np.array(log_data["val_loss"]),
        )

if __name__ == '__main__':
    for model in ['mlp','lstm_mlp']:#,'lstm_cnn']:
        training(model)