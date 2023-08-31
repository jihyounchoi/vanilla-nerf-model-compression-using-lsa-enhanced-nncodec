import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
# D : Depth of MLP
# W : Number of neurons of each layer
# input_ch : xyz coordinate
# input_ch_views : 
# output_ch : RGB + sigma
# skips : layer where skip-connection occur
# use_viewdirs : processing view direction

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        ''' nn.ModuleList
        Similar as nn.Sequential()
        Wrap list of nn.Module, like nn.Sequential
        But there is no forward() method, and module connection between elements.
        '''
        
        # first layer : nn.Linear(input_ch, W)
        # and constructs layers until sigma(density) output, with regarding skip input
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        # ## Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # constructs layer after sigma, e.g. 256 -> 128
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        # ## Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        # constructs final layer
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    
    

############################### CUSTOM ADDED ##################################
class NeRFWrapper(nn.Module):
    
    def __init__(self, D=8, W=256,
                 input_ch=63, input_ch_views=27, 
                 output_ch=4, skips=[4], 
                 use_viewdirs=True):
        
        super(NeRFWrapper, self).__init__()
        
        self.model = NeRF(D=D, W=W, 
                          input_ch=input_ch, input_ch_views=input_ch_views, 
                          output_ch=output_ch, skips=skips, 
                          use_viewdirs=use_viewdirs)
        
        self.model_fine = NeRF(D=D, W=W, 
                               input_ch=input_ch, input_ch_views=input_ch_views, 
                               output_ch=output_ch, skips=skips, 
                               use_viewdirs=use_viewdirs)
        
        # Will automatically assigned at 
        # pytorch_model/__init__.BlenderNeRFModelExecuter.tune_model part
        self.tuning_optimizer = None
        self.global_step = 0
    

def convert_nerf_ckpt_to_nerfwrapper(ckpt_path, D=8, W=256,
                                     input_ch=63, input_ch_views=27, 
                                     output_ch=4, skips=[4], 
                                     use_viewdirs=True):
    # Load the checkpoint
    if not torch.cuda.is_available():
        print("TORCH.CUDA.IS_AVAILABLE() --> FALSE")
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(ckpt_path)
    
    nerf_wrapper = NeRFWrapper(D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs)

    # Extract the model state dictionaries
    network_fn_state_dict = checkpoint['network_fn_state_dict']
    network_fine_state_dict = checkpoint['network_fine_state_dict']

    # Load the state dictionaries into the NeRFWrapper model
    nerf_wrapper.model.load_state_dict(network_fn_state_dict)
    nerf_wrapper.model_fine.load_state_dict(network_fine_state_dict)

    return nerf_wrapper


def convert_nerfwrapper_to_nerf_ckpt(nerfwrapper_path, ckpt_path):
    
    loaded_ckpt = torch.load(nerfwrapper_path)

    # Extract state dictionaries for model and model_fine from the loaded checkpoint
    model_state_dict = {k.replace('model.', ''): v for k, v in loaded_ckpt.items() if 'model.' in k}
    model_fine_state_dict = {k.replace('model_fine.', ''): v for k, v in loaded_ckpt.items() if 'model_fine.' in k}

    # Create a list of all model parameters from the state dictionaries
    grad_vars = list(model_state_dict.values()) + list(model_fine_state_dict.values())

    # Initialize the Adam optimizer with the provided learning rate
    optimizer = torch.optim.Adam(params=grad_vars, lr=0.0001, betas=(0.9, 0.999))

    # Create a new checkpoint in the desired format
    nerf_ckpt = {
        'network_fn_state_dict': model_state_dict,
        'network_fine_state_dict': model_fine_state_dict,
        'global_step': 200000,
        'optimizer_state_dict': optimizer.state_dict()
    }

    torch.save(nerf_ckpt, ckpt_path)

    print(f"Saved the checkpoint in standard nerf_ckpt format to {ckpt_path}")


def change_extension_to_tar(model_path: str) -> str:
    """
    Change the extension of the given model path to .tar.

    Args:
    - model_path (str): Path to the model file.

    Returns:
    - str: Model path with .tar extension.
    """
    # Split the path into root and extension
    root, _ = os.path.splitext(model_path)
    
    # Return the path with .tar extension
    return root + ".tar"


def convert_tar_to_pt(tar_file_path, pt_file_path):
    """
    # Example usage:
        tar_file_path = '/home/gbang/jihyoun/NeRF/nerf-pytorch/logs/blender_paper_lego/200000.tar'
        pt_file_path = './saved_model_lego.pt'

        convert_tar_to_pt(tar_file_path, pt_file_path)
    """
    
    # Load the .tar file
    checkpoint = torch.load(tar_file_path)
    
    print(checkpoint.keys())
    
    # Extract the model state dictionaries
    global_step = checkpoint['global_step']
    network_fn_state_dict = checkpoint['network_fn_state_dict']
    network_fine_state_dict = checkpoint['network_fine_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']

    # Save the model state dictionaries to the .pt file
    torch.save({
        'global_step' : global_step,
        'network_fn_state_dict': network_fn_state_dict,
        'network_fine_state_dict': network_fine_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
    }, pt_file_path)
    
    

def create_save_path(base_path_to_save: str,
                     ckpt_nickname: str,
                     qp: int,
                     lsa: bool,
                     epochs : int,
                     learning_rate : int,
                     task_type : str,
                     dataset_type : str,
                     N_iters : int,
                     learning_rate_decay : float) -> dict:

    now = datetime.now()
    time_minus_9_hours = now - timedelta(hours=9)
    current_time = time_minus_9_hours.strftime('%y%m%d%H%M%S')
    
    filename = os.path.splitext(os.path.basename(ckpt_nickname))[0]

    if lsa:
        info_str = f"{current_time}_{filename}_qp{qp}_e{epochs}_lr{str(learning_rate).replace('.', 'p')}_decay{learning_rate_decay}_N{N_iters}_{dataset_type}"
    else:
        info_str = f"{current_time}_lsaFalse_{filename}_qp{qp}_{dataset_type}"

    bitstream_dir = os.path.join(base_path_to_save, info_str, 'bitstream')
    reconstructed_dir = os.path.join(base_path_to_save, info_str, 'reconstructed')

    # Makes directory if not exists
    os.makedirs(bitstream_dir, exist_ok=True)
    os.makedirs(reconstructed_dir, exist_ok=True)

    bitstream_path = os.path.join(bitstream_dir, f'{info_str}_bitstream.nnc')
    reconstructed_path = os.path.join(reconstructed_dir, f'{info_str}_reconstructed.pt')

    return {'bitstream': bitstream_path, 'reconstructed': reconstructed_path}



if __name__ == '__main__':
    
    result = [(lambda like_backjun : print('crazy') if like_backjun else print('normal'))(like_backjun) for like_backjun in [1, 0, "", None]]; print(result)