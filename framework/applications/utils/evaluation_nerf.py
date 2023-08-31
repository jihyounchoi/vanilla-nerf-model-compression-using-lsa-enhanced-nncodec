########################################### FOR NeRF_Classification_model ###################################################

########################## CUSTOM ADDED ######################

### Dummy Model ###
CALL_TIME = 0
PSNR_LIST = [30, 32, 33, 33.5, 33.6, 33.65, 33.655, 33.656, 33.656]
LOSS_LIST = [0.5, 0.3, 0.2, 0.15, 0.14, 0.135, 0.134, 0.13399, 0.13399]

def evaluate_nerf_model(model, device, mode = 'infinite'):
    """
    Mock version
    """
    
    global CALL_TIME
    psnr, ssim, loss = 0, 0, 0
    
    if mode == 'standard':
        pass
    
    elif mode == 'finite':
        psnr = PSNR_LIST[CALL_TIME]
        ssim = 0.9 # currently not used
        loss = LOSS_LIST[CALL_TIME]
            
    elif mode == 'infinite':
        psnr = CALL_TIME * 10
        ssim = CALL_TIME * 0.1
        loss = CALL_TIME
    
    else:
        raise ValueError(f"Wrong Mode! : {mode}")
    
    CALL_TIME += 1
    
    return psnr, ssim, loss


#############################################################