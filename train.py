import torch

def sobolev_transform(x, c=5, s=1):
    x_fft = torch.fft(torch.stack([x, torch.zeros_like(x)], -1), signal_ndim=2)
    # x_fft[..., 0] stands for real part
    # x_fft[..., 1] stands for imaginary part

    dx = x_fft.shape[3]
    dy = x_fft.shape[2]
    
    x = torch.range(0, dx - 1)
    x = torch.min(x, dx - x)
    x = x / (dx // 2)
    
    y = torch.range(0, dy - 1)
    y = torch.min(y, dy - y)
    y = y / (dy // 2)

    # constructing the \xi domain    
    X, Y = torch.meshgrid([y, x])
    X = X[None, None]
    Y = Y[None, None]
    
    # computing the scale (1 + |\xi|^2)^{s/2}
    scale = (1 + c * (X**2 + Y**2))**(s/2)

    # scale is a real number which scales both real and imaginary parts by multiplying
    scale = torch.stack([scale, scale], -1)
    
    x_fft *= scale.double()
    
    res = torch.ifft(x_fft, signal_ndim=2)[..., 0]
    
    return res

