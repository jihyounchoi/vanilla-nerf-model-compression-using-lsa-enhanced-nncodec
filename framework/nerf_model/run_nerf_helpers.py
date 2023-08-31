import torch
import os
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##############################################################################################

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        
        # Key part to generate positional encoding (gamma(p) = (sin, cos, sin, cos, ...))
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']: # periodic_fns = sin, cos
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Ray helpers
def get_rays(H, W, K, c2w):
    
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    
    i = i.t()
    j = j.t()

    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True) # Normalize weights to make pdf : sum of weights = 1
    
    # Get cdf
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

##################################################### CUSTOM ADDED ########################################################
def device_checker(vars_dict):
    
    device_set = set()
    
    for name, variable in vars_dict.items():
        
        if not isinstance(variable, torch.Tensor):
            print(f"Variable {name} is type {type(variable)}")
            continue
            
        device_set.add(variable.device)
        print(f'Tensor {name} is on {variable.device}')

    if len(device_set) > 1:
        print(f"At least two devices are founded : {device_set}")
    else:
        print("All tensors are on the same device.")
        

def append_results_to_file(basedir, psnr_value, loss_value):
    import os
    
    result_file = os.path.join(basedir, 'result.txt')
    
    # If file doesn't exist, create and initialize it
    if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
        with open(result_file, 'w') as f:
            f.write("psnr : []\n")
            f.write("loss : []\n")
    
    with open(result_file, 'r') as f:
        lines = f.readlines()

    # Extract and modify the psnr line
    psnr_line = lines[0]
    insert_pos = psnr_line.rfind(']')
    new_psnr_line = psnr_line[:insert_pos] + (", " if insert_pos > 8 else "") + f"{psnr_value:.5f}]" + "\n"

    # Extract and modify the loss line
    loss_line = lines[1]
    insert_pos = loss_line.rfind(']')
    new_loss_line = loss_line[:insert_pos] + (", " if insert_pos > 8 else "") + f"{loss_value:.6f}]" + "\n"

    # Write modified lines back to the file
    with open(result_file, 'w') as f:
        f.write(new_psnr_line)
        f.write(new_loss_line)
        
def are_models_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def print_optimizer_state(optimizer):
    state_dict = optimizer.state_dict()

    print("=== Optimizer State ===")
    
    # Print general details
    print(f"Optimizer: {optimizer.__class__.__name__}")

    # Print param groups
    print("\nParameter Groups:")
    for i, param_group in enumerate(state_dict['param_groups']):
        print(f"\nParam Group {i}:")
        for key, value in param_group.items():
            if key != 'params':
                print(f"  {key}: {value}")

    # Print detailed state
    print("\nDetailed State:")
    for param_id, param_state in state_dict['state'].items():
        print(f"\nParam ID {param_id}:")
        for key, value in param_state.items():
            # We print the shape for tensor objects to avoid overwhelming the output
            if isinstance(value, torch.Tensor):
                print(f"  {key}: tensor of shape {value.shape}")
            else:
                print(f"  {key}: {value}")
###########################################################################################################################