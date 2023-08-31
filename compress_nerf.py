import argparse
import utils
import nnc

def main(args):
    # Use the parsed arguments
    ckpt_path = args.ckpt_path
    ckpt_nickname = args.ckpt_nickname
    base_path_to_save = args.base_path_to_save
    qp = args.qp
    lsa = args.lsa
    epochs = args.epochs
    learning_rate = args.learning_rate
    task_type = args.task_type
    dataset_type = args.dataset_type
    N_iters = args.N_iters
    learning_rate_decay = args.learning_rate_decay
    i_save = args.i_save
    dataset_path = args.dataset_path

    # 1. Create NeRFWrapper Instance
    nerf_wrapper = utils.convert_nerf_ckpt_to_nerfwrapper(ckpt_path=ckpt_path)
    
    
    # 2. Create Proper Name of Bitstream, and Reconstructed Model
    path_dict = utils.create_save_path( ckpt_nickname=ckpt_nickname,
                                        base_path_to_save=base_path_to_save,
                                        qp=qp,
                                        lsa=lsa,
                                        epochs=epochs,
                                        learning_rate=learning_rate,
                                        task_type=task_type,
                                        dataset_type=dataset_type,
                                        N_iters=N_iters,
                                        learning_rate_decay=learning_rate_decay)


    # 3. Compress NeRFWrapper Model
    nnc.compress_model( model_path_or_object=nerf_wrapper, 
                        bitstream_path=path_dict['bitstream'], 
                        qp=qp,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        lsa=lsa, 
                        model_struct=nerf_wrapper,
                        dataset_path=dataset_path,
                        task_type=task_type,
                        dataset_type=dataset_type,
                        N_iters = N_iters, 
                        learning_rate_decay = learning_rate_decay, 
                        i_save = i_save
                        )



    # 4. Decompress Model
    nnc.decompress_model(path_dict['bitstream'], 
                         model_path=path_dict['reconstructed'])
    
    
    # 5. Convert Decompress NeRF-Wrapper to Standard NeRF Checkpoint (.tar)
    utils.convert_nerfwrapper_to_nerf_ckpt(nerfwrapper_path=path_dict['reconstructed'], 
                                           ckpt_path=utils.change_extension_to_tar(path_dict['reconstructed']))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NeRF Processing Script")
    
    # Common Parameters
    parser.add_argument('--ckpt_path', type=str, default="/home/gbang/jihyoun/NeRF/nerf_lsa/model_zoo/blender_paper_lego/lego_200000.tar", help="Path to checkpoint file.")
    parser.add_argument('--ckpt_nickname', default = 'lego_200K', type=str, help="Nickname Of Checkpoint, like lego-200000, ...")
    parser.add_argument('--base_path_to_save', type=str, default='/home/gbang/jihyoun/NeRF/nerf_lsa', help="Base path to save.")
    parser.add_argument('--qp', type=int, default=-15, help="Quantization Parameter.")
    
    # LSA-Related Parameters
    parser.add_argument('--lsa', type=bool, default=True, help="Use LSA or not.") # Should be True
    parser.add_argument('--epochs', type=int, default=2, help="Number of epochs.")
    parser.add_argument('--learning_rate', type=float, default = 0.0001, help="Learning rate.")
    
    # Added Parameters for NeRF-LSA
    parser.add_argument('--task_type', type=str, default='NeRF', help="Task type. Classification or NeRF Available") # Classification or NeRF
    parser.add_argument('--dataset_type', type=str, default='blender', help="Dataset type.") # blender or llff
    parser.add_argument('--N_iters', type=int, default = 20000, help="Number of iterations.") # Number of iteration of each epoch
    parser.add_argument('--learning_rate_decay', type=float, default=0.5, help="Learning rate decay.") # learning rate decay after each epoch finished
    parser.add_argument('--i_save', type=int, default=10000, help="Interval for saving.") # Saving interval (iterations)
    
    # Only Necessary for Classification Task (NOT for NeRF)
    parser.add_argument('--dataset_path', type=str, default='~', help="Path to dataset.")
    
    args = parser.parse_args()
    
    expected_types = {
        'ckpt_path': str,
        'ckpt_nickname': str,
        'base_path_to_save': str,
        'qp': int,
        'lsa': bool,
        'epochs': int,
        'learning_rate': float,
        'task_type': str,
        'dataset_type': str,
        'N_iters': int,
        'learning_rate_decay': float,
        'i_save': int,
        'dataset_path': str
    }

    for arg, expected_type in expected_types.items():
        if not isinstance(getattr(args, arg), expected_type):
            raise ValueError(f"Argument {arg} is not of expected type {expected_type}. It is of type {type(getattr(args, arg))}.")
    
    print("\n")
    print("############## PROVIDED ARGUMENTS ################")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("##################################################")
    print("\n")
    
    main(args)