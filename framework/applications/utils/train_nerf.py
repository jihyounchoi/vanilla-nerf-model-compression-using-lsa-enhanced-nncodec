import torch
import os

from framework.nerf_model import run_nerf

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def freeze_batch_norm_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


def train_nerf_model(nerf_wrapper,
                     dataset_type,
                     freeze_batch_norm,
                     basedir_save,
                     N_iters, 
                     i_save
                     ):


    nerf_wrapper.model.to(DEVICE)
    nerf_wrapper.model_fine.to(DEVICE)
    
    # if freeze_batch_norm:
    #     freeze_batch_norm_layers(nerf_wrapper)
        
    psnr, loss = 0, 0
    
    if dataset_type == 'blender':

        # Precrop_iters was originally 500.
        # But in the case of training lsa parameter of pretrained nerf model, it is set to 0
        
        
        psnr, loss = run_nerf.train(nerf_wrapper=nerf_wrapper, 
                                    basedir_save = basedir_save,
                                    basedir=os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../nerf_model/logs'),
                                    datadir=os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../nerf_model/data/nerf_synthetic/lego'),
                                    dataset_type="blender",
                                    no_batching=True,
                                    use_viewdirs=True,
                                    white_bkgd=True,
                                    N_samples=64,
                                    N_importance=128,
                                    N_rand=1024,
                                    precrop_iters=0,  
                                    precrop_frac=0.5,
                                    half_res=True,
                                    N_iters=N_iters, 
                                    i_save=i_save
                                    )
        
    elif dataset_type == 'llff':
        psnr, loss = run_nerf.train(nerf_wrapper=nerf_wrapper,
                                    basedir_save = basedir_save,
                                    basedir=os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../nerf_model/logs'),
                                    datadir=os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../nerf_model/data/nerf_llff_data/fern'),
                                    dataset_type="llff",
                                    factor=8,
                                    llffhold=8,
                                    N_rand=1024,
                                    N_samples=64,
                                    N_importance=64,
                                    use_viewdirs=True,
                                    raw_noise_std=1e0,
                                    N_iters=N_iters, 
                                    i_save=i_save
                                    )
    else:
        raise ValueError(f"dataset_type : {dataset_type} is not yet implemented, or wrong!")
    
    return psnr, loss