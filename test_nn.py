import numpy as np
import torch
import time
import os
import glob

from classes import WavesDataset, MLP, LSTM_MLP, LSTM_CNN

def testing(name):
    timeenc_dim = 2
    lead_times = [0, 8, 16, 24]

    ds = WavesDataset('test', name, False, timeenc_dim, '', lead_times)

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

    train_size = 0
    val_size = ds.__len__()
    batch_size = 2 ** 21
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    name = name + '_timeenc'
    weights_dir = glob.glob(f'weights/{name}/{name}_epoch_*.pth')
    latest_weights = max(weights_dir, key=lambda f: int(f.split('_epoch_')[-1].split('.pth')[0]))
    print(f'Loading weights: {latest_weights}')
    weights = torch.load(latest_weights)
    model.load_state_dict(weights['model_state_dict'])
    optimizer.load_state_dict(weights['optimizer_state_dict'])
    start_epoch = weights['epoch']
    print(f'Resuming from {start_epoch}th epoch')
    os.makedirs("logs/advs2/", exist_ok=True)

    model.eval()
    with torch.no_grad():
        ds.sort_dt() # sort lead times for validation
        ds.ts_dt()  # complete lead times for adv log
        for jdx in range(max(lead_times) + 1):
            adv_log_data = -1 * torch.ones([1, 9], device='cuda')
            for i in range(train_size, train_size + val_size - lead_times[-1] * ds.inputs['ratio'],
                           min(batch_size, train_size + val_size - lead_times[-1] * ds.inputs['ratio'])):
                idx = torch.arange(i,
                                   min(i + batch_size, train_size + val_size - lead_times[-1] * ds.inputs['ratio']),
                                   device='cuda')
                adv_log_data = torch.cat([adv_log_data, torch.cat(
                    [ds.targets['t'][idx].repeat_interleave(168 // ds.inputs['ratio']).reshape(-1, 1),
                     model.get_adv_log(ds.inputs, ds.targets, idx, jdx)],
                    dim=1)], dim=0)
            np.savez_compressed(f"logs/advs2/adv_log-{name}-{ds.inputs['dt'][0, :][jdx]}h.npz",
                                adv_log_data[1:, :].detach().cpu().numpy())

if __name__ == '__main__':
    for model in ['mlp','lstm_mlp','lstm_cnn']:
        testing(model)