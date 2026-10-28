import numpy as np
import torch

from piml_aux import energy

class WavesDataset(torch.utils.data.Dataset):
    def __init__(self, file, model, piml, time_enc, mask, lead_times):
        file = file + '_timeenc' if time_enc else file
        path = 'data/' + file + '.npz'
        data = np.load(path)
        self.inputs = {}
        self.targets = {}
        
        if model=='lstm_mlp' or model=='lstm_cnn':
            input_dim = 3 # Tm, Hscos, Hssin
            b = data['buoys_coords'].shape[0]
            l = data['buoys'].shape[0]
            n = (data['buoys'].shape[1]-1-time_enc)//b
            s = n//input_dim
            inputs = np.empty((b,l,s,2+input_dim))
            for i in range(1,b+1):
                rep_coords = np.repeat(np.repeat(data['buoys_coords'][i-1,:].reshape(1,2),l,axis=0),s,axis=0).reshape(l,s,2)
                seq_features = data['buoys'][:,(time_enc+(1+(i-1)*n)):(time_enc+(1+i*n))].reshape(l,s,input_dim)
                inputs[i-1,:,:,:] = np.concatenate((rep_coords, seq_features), axis=2)
            self.inputs['buoys'] = torch.from_numpy(inputs).float().to('cuda')
        else: # mode == 'mlp'
            self.inputs['buoys'] = torch.from_numpy(data['buoys'][:,(1+time_enc):]).float().to('cuda')
        
        if model != 'lstm_cnn':
            if time_enc:
                self.inputs['coords'] = torch.cat((torch.from_numpy(data['label'][:,(1+time_enc):(3+time_enc)]).float().to('cuda'),
                                                   torch.from_numpy(data['label'][:,1:(1+time_enc)]).float().to('cuda')), axis=1)
            else:
                self.inputs['coords'] = torch.from_numpy(data['label'][:,1:3]).float().to('cuda')
            self.inputs['ratio'] = torch.tensor(len(data['label'])/len(data['buoys'])).to(torch.long).to('cuda')
            self.targets['label'] = torch.from_numpy(data['label'][:,(3+time_enc):]).float().to('cuda')

            self.targets['t'] = torch.from_numpy(data['label'][:, 0]).float().to('cuda')
        else: # mode == 'lstm_cnn'
            if time_enc:
                self.inputs['time_enc'] = torch.from_numpy(data['buoys'][:,1:1+time_enc]).float().to('cuda')
            else:
                self.inputs['time_enc'] = torch.empty((len(data['label']),0),device='cuda')
            self.inputs['ratio'] = torch.tensor(len(data['label_cnn'])/len(data['buoys'])).to(torch.long).to('cuda') # ==1, for idx consistency
            label = data['label_cnn'][:,(1+time_enc):]
            targets = np.where(label==-10, np.nan, label).reshape(-1, 14, 17, 3)
            self.targets['label'] = torch.from_numpy(targets).float().to('cuda')
            self.targets['cnn_coords'] = torch.from_numpy(data['cnn_coords']).float().to('cuda')
            self.targets['cnn_mask'] = torch.from_numpy(~np.isnan(targets)).to('cuda')

            self.targets['t'] = torch.from_numpy(data['label_cnn'][:, 0]).float().to('cuda')

        lead_times_tensor = torch.tensor(lead_times).to('cuda')
        self.inputs['dt'] = lead_times_tensor[torch.rand(self.targets['label'].shape[0], lead_times_tensor.shape[0]).argsort(dim=1)]

        if piml:
            stats = np.load('data/statistics.npz')
            self.inputs['stats'] = torch.tensor(np.array([[stats['Tm_mean'],    stats['Tm_std']],
                                                           [stats['Hscos_mean'], stats['Hscos_std']],
                                                           [stats['Hssin_mean'], stats['Hssin_std']]])).to('cuda')
            #[0,                       3600],
            #[stats['latitude_mean'],  stats['latitude_std']],
            #[stats['longitude_mean'], stats['longitude_std']],
            if model == 'lstm_cnn':
                self.inputs['pi_mask'] = torch.from_numpy(np.load('data/masks.npz')['piml_cnn']).to('cuda')
            else:
                self.inputs['pi_mask'] = torch.from_numpy(np.load('data/masks.npz')['piml']).to('cuda').repeat(self.targets['label'].shape[0]//
                                                                                                                self.inputs['ratio'])
        if model == 'lstm_cnn':
            self.targets['no_mask'] = torch.ones(self.targets['label'].shape, device='cuda').to(torch.bool)
        else:
            self.targets['no_mask'] = torch.ones(self.targets['label'].shape, device='cuda').to(torch.bool)
        if mask:
            if model == 'lstm_cnn':
                map = torch.from_numpy(np.stack([np.concatenate([np.zeros(17 - len(c), dtype=bool), c])
                                for c in np.split(np.load('data/masks.npz')[mask],
                                                  np.cumsum([17,15,15,15,15,15,15,15,14,12,10,7,2,1])[:-1])])).to('cuda')
                self.targets['loaded_mask'] = map.reshape(1,14,17,1).repeat(self.targets['label'].shape[0],1,1,self.targets['label'].shape[3])
            else:
                self.targets['loaded_mask'] = torch.from_numpy(np.load('data/masks.npz')[mask]).repeat(self.targets['label'].shape[0]//
                                                                                                self.inputs['ratio']).reshape(-1,1).repeat(1,3).to('cuda')
            self.targets['mask'] = self.targets['loaded_mask']
        else:
            self.targets['mask'] = self.targets['no_mask']

    def shuffle_dt(self):
        lead_times_tensor = self.inputs['dt'][0,:]
        self.inputs['dt'] = lead_times_tensor[torch.rand(self.targets['label'].shape[0], lead_times_tensor.shape[0]).argsort(dim=1)]

    def sort_dt(self):
        lead_times_tensor = self.inputs['dt']
        self.inputs['dt'] = lead_times_tensor.sort(dim=1).values

    def ts_dt(self):
        lead_times_tensor = torch.arange(self.inputs['dt'].max()+1).to('cuda')
        self.inputs['dt'] = lead_times_tensor.repeat(self.targets['label'].shape[0],1)

    def enable_mask(self):
        self.targets['mask'] = self.targets['loaded_mask']

    def disable_mask(self):
        self.targets['mask'] = self.targets['no_mask']

    def __len__(self):
        return max(self.targets['label'].shape)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        target_sample = self.targets[idx]
        return input_sample, target_sample
    
class MLP(torch.nn.Module):
    def __init__(self, num_layers, width, activ_fun, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torch.nn.Sequential()
        self.model.add_module('fc1', torch.nn.Linear(input_dim, width, bias=False))
        self.model.add_module('bn1', torch.nn.BatchNorm1d(width))
        self.model.add_module('activ1', activ_fun())
        for i in range(2, num_layers+1):
            self.model.add_module(f'fc{i}', torch.nn.Linear(width, width, bias=False))
            self.model.add_module(f'bn{i}', torch.nn.BatchNorm1d(width))
            self.model.add_module(f'activ{i}', activ_fun())
        self.model.add_module('output', torch.nn.Linear(width, output_dim))

    def forward(self, inputs):
        return self.model(inputs)
    
    def loss(self, inputs, targets, idx, jdx):
        loss = (((self(torch.cat((inputs['dt'][idx,jdx].reshape(-1,1)/24,
                                 inputs['coords'][idx],
                                 inputs['buoys'][torch.floor_divide(idx, inputs['ratio'])]),
                                dim=1))
                 -targets['label'][idx+inputs['dt'][idx,jdx]*inputs['ratio']])**2)
                 [targets['mask'][idx+inputs['dt'][idx,jdx]*inputs['ratio']]].reshape(-1,3)
                 .mean(axis=0))
        return loss.mean(), loss[0], loss[1], loss[2]
    
    def swell_loss(self, inputs, idx, jdx):
        dt = 1 #h
        dlat = 0.5 #deg
        dlon = 0.5 #deg
        Dlat = torch.cat((dlat/inputs['stats'][1,1]*torch.ones(inputs['coords'][idx].shape[0],1,device='cuda'),
                          torch.zeros(inputs['coords'][idx].shape[0],inputs['coords'][idx].shape[1]-1,device='cuda')),axis=1)
        Dlon = torch.cat((torch.zeros(inputs['coords'][idx].shape[0],1,device='cuda'),
                          dlon/inputs['stats'][2,1]*torch.ones(inputs['coords'][idx].shape[0],1,device='cuda'),
                          torch.zeros(inputs['coords'][idx].shape[0],inputs['coords'][idx].shape[1]-2,device='cuda')),axis=1)
        dt0 = inputs['dt'][idx, jdx].reshape(-1, 1) / 24
        buoys = inputs['buoys'][torch.floor_divide(idx, inputs['ratio'])]
        coords = inputs['coords'][idx]
        preds = [
            self(torch.cat((dt0,         coords,        buoys), dim=1)),
            self(torch.cat((dt0 + dt/24, coords,        buoys), dim=1)),
            self(torch.cat((dt0,         coords + Dlat, buoys), dim=1)),
            self(torch.cat((dt0,         coords + Dlon, buoys), dim=1))]

        res = energy(dt, dlat, dlon, preds, inputs['stats'])
        return (res[inputs['pi_mask'][idx]]**2).mean()
    
    def pi_loss(self, inputs, targets, idx, jdx):
        data_losses = self.loss(inputs, targets, idx, jdx)
        phy_loss = self.swell_loss(inputs, idx, jdx)
        return phy_loss*1e1+data_losses[0], data_losses[1], data_losses[2], data_losses[3]

    def get_adv_log(self, inputs, targets, idx, jdx):
        pred = self(torch.cat((inputs['dt'][idx,jdx].reshape(-1,1)/24,
                               inputs['coords'][idx],
                               inputs['buoys'][torch.floor_divide(idx, inputs['ratio'])]),
                              dim=1))
        label = targets['label'][idx+inputs['dt'][idx,jdx]*inputs['ratio']]
        return torch.cat((inputs['coords'][idx,:2], pred, label), dim=1)
    
class LSTM_encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        #self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        last_hidden_state = lstm_out[:, -1, :]
        #output = self.fc(last_hidden_state)
        return last_hidden_state#output
    
class CNN_decoder(torch.nn.Module):
    def __init__(self, activ_fun, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = torch.nn.Linear(input_dim, 2**6) # hardcoded for 25-35 input size, 14,17 output size and 3 output channels
        self.cnn = torch.nn.Sequential()
        for i in range(3):
            self.cnn.add_module(f'fc{i+1}', torch.nn.ConvTranspose2d(2**(6-i), 2**(6-(1+i)), kernel_size=4, stride=2, padding=1))
            self.cnn.add_module(f'activ{i+1}', activ_fun())
        self.cnn.add_module(f'fc4', torch.nn.ConvTranspose2d(2**3, output_dim, kernel_size=4, stride=2, padding=1, output_padding=1))
        self.cnn.add_module(f'activ4', activ_fun())

    def forward(self, inputs):
        prev = self.fc(inputs).reshape(-1, self.cnn[0].in_channels, 1, 1)
        output = self.cnn(prev)[:, :, :14, :17].swapaxes(1, 2).swapaxes(2, 3)
        return output

class LSTM_MLP(torch.nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, lstm_num_layers, lstm_output_dim,
                 mlp_add_input_dim, mlp_num_layers, mlp_width, mlp_activ_fun,
                 output_dim, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm_encoder = LSTM_encoder(lstm_input_dim, lstm_hidden_dim,
                                         lstm_num_layers, lstm_output_dim).to(device)
        self.mlp = MLP(mlp_num_layers, mlp_width, mlp_activ_fun, mlp_add_input_dim+3*lstm_output_dim, output_dim).to(device)

    def forward(self, inputs_lstm, inputs_add_mlp):
        lstm_outs = torch.empty((inputs_lstm.shape[0], inputs_lstm.shape[1], self.lstm_encoder.lstm.hidden_size), device='cuda') #self.lstm_encoder.fc.out_features
        for i in range(len(inputs_lstm)):
            lstm_outs[i,:,:] = self.lstm_encoder(inputs_lstm[i,:,:,:])
        lstm_out = torch.cat((inputs_add_mlp, lstm_outs.swapaxes(0, 1).reshape(inputs_lstm.shape[1], -1)), axis=1)
        output = self.mlp(lstm_out)
        return output
    
    def loss(self, inputs, targets, idx, jdx):
        loss = (((self(inputs['buoys'][:,torch.floor_divide(idx, inputs['ratio']),:,:],
                      torch.cat((inputs['dt'][idx,jdx].reshape(-1,1)/24,inputs['coords'][idx]), dim=1))
                 -targets['label'][idx+inputs['dt'][idx,jdx]*inputs['ratio']])**2)
                 [targets['mask'][idx+inputs['dt'][idx,jdx]*inputs['ratio']]].reshape(-1,3).mean(axis=0))
        return loss.mean(), loss[0], loss[1], loss[2]
    
    def swell_loss(self, inputs, idx, jdx):
        dt = 1 #h
        dlat = 0.5 #deg
        dlon = 0.5 #deg
        Dlat = torch.cat((dlat/inputs['stats'][1,1]*torch.ones(inputs['coords'][idx].shape[0],1,device='cuda'),
                          torch.zeros(inputs['coords'][idx].shape[0],inputs['coords'][idx].shape[1]-1,device='cuda')),axis=1)
        Dlon = torch.cat((torch.zeros(inputs['coords'][idx].shape[0],1,device='cuda'),
                          dlon/inputs['stats'][2,1]*torch.ones(inputs['coords'][idx].shape[0],1,device='cuda'),
                          torch.zeros(inputs['coords'][idx].shape[0],inputs['coords'][idx].shape[1]-2,device='cuda')),axis=1)
        buoys = inputs['buoys'][:,torch.floor_divide(idx, inputs['ratio']),:,:]
        dt0 = inputs['dt'][idx, jdx].reshape(-1, 1) / 24
        coords = inputs['coords'][idx]
        preds = [
            self(buoys,torch.cat((dt0,         coords       ), dim=1)),
            self(buoys,torch.cat((dt0 + dt/24, coords       ), dim=1)),
            self(buoys,torch.cat((dt0,         coords + Dlat), dim=1)),
            self(buoys,torch.cat((dt0,         coords + Dlon), dim=1))]
        res = energy(dt, dlat, dlon, preds, inputs['stats'])
        return (res[inputs['pi_mask'][idx]]**2).mean()
    
    def pi_loss(self, inputs, targets, idx, jdx):
        data_losses = self.loss(inputs, targets, idx, jdx)
        phy_loss  = self.swell_loss(inputs, idx, jdx)
        return phy_loss*1e1+data_losses[0], data_losses[1], data_losses[2], data_losses[3]

    def get_adv_log(self, inputs, targets, idx, jdx):
        pred = self(inputs['buoys'][:,torch.floor_divide(idx, inputs['ratio']),:,:],
                    torch.cat((inputs['dt'][idx,jdx].reshape(-1,1)/24,inputs['coords'][idx]), dim=1))
        label = targets['label'][idx+inputs['dt'][idx,jdx]*inputs['ratio']]
        return torch.cat((inputs['coords'][idx,:2], pred, label), dim=1)
    
class LSTM_CNN(torch.nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, lstm_num_layers, lstm_output_dim,
                 cnn_add_input_dim, cnn_activ_fun, output_dim, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm_encoder = LSTM_encoder(lstm_input_dim, lstm_hidden_dim,
                                         lstm_num_layers, lstm_output_dim).to(device)
        self.cnn_decoder = CNN_decoder(cnn_activ_fun, cnn_add_input_dim+3*lstm_output_dim, output_dim).to(device)
        
    def forward(self, inputs_lstm, inputs_add_cnn):
        lstm_outs = torch.empty((inputs_lstm.shape[0], inputs_lstm.shape[1], self.lstm_encoder.lstm.hidden_size), device='cuda') #self.lstm_encoder.fc.out_features
        for i in range(len(inputs_lstm)):
            lstm_outs[i,:,:] = self.lstm_encoder(inputs_lstm[i,:,:,:])
        lstm_out = torch.cat((inputs_add_cnn, lstm_outs.swapaxes(0, 1).reshape(inputs_lstm.shape[1], -1)), axis=1)
        output = self.cnn_decoder(lstm_out)
        return output

    def loss(self, inputs, targets, idx, jdx):
        pred = self(inputs['buoys'][:,idx,:,:],
                    torch.cat((inputs['dt'][idx,jdx].reshape(-1,1)/24,inputs['time_enc'][idx]),axis=1))
        label = targets['label'][idx+inputs['dt'][idx,jdx]]#*inputs['ratio']]
        mask = targets['cnn_mask'][idx+inputs['dt'][idx,jdx]]*targets['mask'][idx+inputs['dt'][idx,jdx]]
        loss = ((pred-label)**2)[mask].reshape(-1, 3).nanmean(axis=0)
        return torch.nn.functional.mse_loss(pred[mask], label[mask]), loss[0], loss[1], loss[2]
    
    def swell_loss(self, inputs, idx, jdx):
        dt = 1 #h
        dlat = 0.5 #deg
        dlon = 0.5 #deg
        pred0 = self(inputs['buoys'][:,idx,:,:],
                     torch.cat((inputs['dt'][idx,jdx].reshape(-1,1)/24,inputs['time_enc'][idx]),axis=1))
        pred1 = self(inputs['buoys'][:,idx,:,:],
                     torch.cat(((inputs['dt'][idx,jdx].reshape(-1,1)+dt)/24,inputs['time_enc'][idx]),axis=1))
        pred2 = pred0.clone()
        pred3 = pred0.clone()
        pred2[:,:-1,:,:] = pred0[:,1:,:,:]
        pred3[:,:,:-1,:] = pred0[:,:,1:,:]
        mask = inputs['pi_mask'].reshape(-1)
        preds = [pred0.reshape(-1, 14*17, 3)[:,mask,:].reshape(-1, 3),
                 pred1.reshape(-1, 14*17, 3)[:,mask,:].reshape(-1, 3),
                 pred2.reshape(-1, 14*17, 3)[:,mask,:].reshape(-1, 3),
                 pred3.reshape(-1, 14*17, 3)[:,mask,:].reshape(-1, 3)]
        
        res = energy(dt, dlat, dlon, preds, inputs['stats'])
        return (res**2).mean()

    def pi_loss(self, inputs, targets, idx, jdx):
        data_losses = self.loss(inputs, targets, idx, jdx)
        phy_loss  = self.swell_loss(inputs, idx, jdx)
        return phy_loss*1e-1+data_losses[0], data_losses[1], data_losses[2], data_losses[3]
    
    def get_adv_log(self, inputs, targets, idx, jdx):
        pred = self(inputs['buoys'][:,idx,:,:],
                    torch.cat((inputs['dt'][idx,jdx].reshape(-1,1)/24,inputs['time_enc'][idx]),axis=1))
        label = targets['label'][idx+inputs['dt'][idx,jdx]]
        data = torch.cat((targets['cnn_coords'].reshape(1,14,17,2).expand(label.shape[0], -1, -1, -1),
                          pred, label), dim=3)
        return data[~torch.isnan(data).any(dim=3)]