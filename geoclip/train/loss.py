import torch


def orientation_loss(pred, target):
    cos_theta = pred[:, 1]
    sin_theta = pred[:, 0]
    angle_rad = torch.atan2(sin_theta, cos_theta)
    pred = torch.rad2deg(angle_rad)

    N = torch.divide(pred, 360)
    pred = torch.where(pred >= 360, pred - (N * 360), pred)
    pred = torch.where(pred < 0, pred + (torch.abs(N) * 360), pred)
    
    error1 = torch.abs(target - pred)
    error2 = torch.sub(360, error1)
    error = torch.min(error1, error2)
    mae = torch.mean(error)
    return mae, error
