import torch

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    autocast = torch.amp.autocast('cuda', dtype=torch.float16)
    grad_scaler = torch.amp.GradScaler('cuda', init_scale=2**12)
    print(f'Use device: {torch.cuda.get_device_name(device)}')
else:
    device = torch.device('cpu')
    autocast = torch.amp.autocast('cpu', dtype=torch.bfloat16)
    grad_scaler = torch.amp.GradScaler('cpu', init_scale=2**16)
    print(f'Use CPU')

