import torch

def denormalize(data, stats):
    denorm = data * stats[:, 1].reshape(1, 1, 3) + stats[:, 0].reshape(1, 1, 3)

    Tm = denorm[:, :, 0]
    Hs = torch.hypot(denorm[:, :, 1], denorm[:, :, 2])
    Dir = torch.atan2(denorm[:, :, 2], denorm[:, :, 1])

    return torch.stack((Tm, Hs, Dir), dim=2)

def energy(dt, dlat, dlon, preds, stats):
    preds = torch.stack(preds, dim=0)
    data = denormalize(preds, stats)
    Tm  = data[:,:,0]
    Hs  = data[:,:,1]
    Dir = data[:,:,2]
    E = (1/16)*1025*9.81*Hs**2 # [J/m^2]
    idx_x = torch.tensor([0, 2], device='cuda')
    idx_y = torch.tensor([0, 3], device='cuda')
    cgx = 1.56 * Tm[idx_x] * torch.cos(Dir[idx_x])
    cgy = 1.56 * Tm[idx_y] * torch.sin(Dir[idx_y])
    delEdelt = (E[1]-E[0])/(dt*3600) # [W/m^2]
    divergcE = -((cgx[1]*E[2]-cgx[0]*E[0])/(dlat*111e3) + (cgy[1]*E[3]-cgy[0]*E[0])/(dlon*101e3)) # [W/m^2] negative 'wave from' to 'wave to'

    return 1e3*(delEdelt + divergcE) # [W/m^2]