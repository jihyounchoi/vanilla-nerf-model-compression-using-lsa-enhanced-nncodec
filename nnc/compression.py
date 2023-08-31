import sys

assert sys.version_info >= (3, 6)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import numpy as np
import copy
import nnc_core

from nnc_core import nnr_model
from framework import tensorflow_model
from framework import pytorch_model
from timeit import default_timer as timer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def __print_output_line( outputString, verbose=True ):
    """
    Helper function to print output strings to the console if verbose is True.
    """
    if verbose:
        sys.stdout.write(outputString)
        sys.stdout.flush()
         
            
def guess_block_id_and_param_type(model_struct, add_lsa_params=False):
    """
    Guesses the block IDs and parameter types for a given model structure.
    Returns the block ID and parameter type dictionary.
    
    Args:
        model_struct: The model structure, can be a TensorFlow or PyTorch model.
        add_lsa_params (bool): Whether to add local scaling adaptation (LSA) parameters.
        
    Returns:
        block_id_and_param_type (dict): The dictionary containing block IDs and parameter types.
    """
    
    if tensorflow_model.is_tef_model(model_struct):
        nnc_mdl, _, model_parameters = tensorflow_model.create_NNC_model_instance_from_object(
                 model_struct,
                )
        block_id_and_param_type = nnc_mdl.guess_block_id_and_param_type(model_parameters)
        
    elif pytorch_model.is_pyt_model(model_struct):
        nnc_mdl, _, model_parameters = pytorch_model.create_NNC_model_instance_from_object(
                 model_struct,
                )
        block_id_and_param_type = nnc_mdl.guess_block_id_and_param_type(model_parameters)
        
    else:
        print("INFO: guess_block_id_and_param_type is only applicable to Pytorch and Tensorflow models! block_id_and_param_type has been set to 'None'")
        block_id_and_param_type=None
        
    if block_id_and_param_type and add_lsa_params:
        for param, parType in block_id_and_param_type["parameter_type"].items():
            if parType == "weight":
                lsa_param = param + "_scaling"
                block_id_and_param_type["parameter_type"][lsa_param] = "weight.ls"
                block_id_and_param_type["block_identifier"][lsa_param] = block_id_and_param_type["block_identifier"][param]
    
    if block_id_and_param_type:    
        blkIdParamTypeOk = nnc_core.nnr_model.sanity_check_block_id_and_param_type( block_id_and_param_type, model_parameters )
        if blkIdParamTypeOk == False:
            print("INFO: Sanity check for block_id_and_param_type failed! block_id_and_param_type has been set to 'None'!")
            block_id_and_param_type = None
    
    return block_id_and_param_type
        

def compress_model( model_path_or_object, # Path to the model file or the model object itself.
                    bitstream_path="./bitstream.nnc", # Path to save the compressed bitstream (default: "./bitstream.nnc").
                    qp=-38, # Quantization parameter for weight approximation (default: -38).
                    qp_density=2, # Density of quantization parameter search space (default: 2).
                    nonweight_qp=-75, # Quantization parameter for non-weight parameters (default: -75).
                    qp_per_tensor=None, # Custom quantization parameters per tensor (default: None).
                    use_dq=True, # Dependent Scalar Quantization (default : True)
                    codebook_mode=0, # Codebook mode for approximation (default: 0).
                    scan_order=0, # Scan order for approximation (default: 0).
                    lambda_scale=0, # Lambda scaling for approximation (default: 0).
                    param_opt=True, # Whether to optimize quantization parameters (default: True).
                    cabac_unary_length_minus1=10, # Length of CABAC unary coder (default: 10).
                    opt_qp=False, # Whether to optimize quantization parameters using RDO (default: False).
                    ioq=False, # Whether to perform inference-based quantization (default: False).
                    bnf=False, # BatchNorm Folding (default : False)
                    lsa=False, # Local Scale Adaptation (default : False)
                    fine_tune=False, # Whether to perform fine-tuning (default: False).
                    block_id_and_param_type=None, # Block ID and parameter type dictionary (default: None).
                    model_name=None, # Name of the model (default: None).
                    model_executer=None, # Model executer object for LSA, FT, and IOQ (default: None).
                    model_struct=None, # Model structure (default: None).
                    dataset_path=None, # Path to the dataset for LSA or fine-tuning (default: None).
                    learning_rate=1e-4, # Learning rate for LSA or fine-tuning (default: 1e-4).
                    batch_size=64, # Batch size for LSA or fine-tuning (default: 64).
                    epochs=100, # Number of epochs for LSA or fine-tuning (default: 30).
                    max_batches=600, # Maximum number of batches for LSA or fine-tuning (default: 600).
                    num_workers=8, # Number of workers for LSA or fine-tuning (default: 8).
                    return_model_data=False, # Whether to return the block ID and parameter type (default: False).
                    verbose=True, # Whether to print verbose output (default: True).
                    return_bitstream=False, # Whether to return the compressed bitstream (default: False).
                    task_type = "Classification",
                    dataset_type = 'blender', # Only Needed When Compressing NeRF with LSA Enabled
                    N_iters = 50000, # N_iter, learning_rate_decay, i_save are only used for NeRF-LSA
                    learning_rate_decay = 0.1, # When set to 0, don't apply lr_decay
                    i_save = 10000
                   ):
    
    """
        Returns:
        Depending on the arguments, the function can return:
        - If `return_model_data` is True: block ID and parameter type dictionary.
        - If `return_bitstream` is True: compressed bitstream.
        - If both `return_model_data` and `return_bitstream` are True: 
          compressed bitstream and block ID and parameter type dictionary.
    """
    
    # Variables to check if the model is a TensorFlow or PyTorch model
    is_pyt_model = False
    is_tef_model = False
    
    # Expand the dataset path if provided
    # Dataset is used for lsa, or ft like dataset-specific operations
    dataset_path = None if dataset_path is None else os.path.expanduser(dataset_path)
    
    # Check if the model is a TensorFlow model
    if tensorflow_model.is_tef_model(model_path_or_object): 
        if bnf:
            print(
                "WARNING: Batch-norm folding (BNF) requires the tensors to be shaped "
                "such that the first dimensions corresponds to the number of output channels, "
                "which is usually not the case for TensorFlow. For further details refer to the Wiki! "
                )
        if lsa:
            print("INFO: LSA not yet supported for TensorFlow models. 'lsa' has been set to false!")
            lsa = False
            
        is_tef_model = True
        
        # Create an NNC model instance from the TensorFlow model object or path
        if model_executer:
                nnc_mdl, _, model_parameters = tensorflow_model.create_NNC_model_instance_from_object(
                 model_path_or_object,
                )
        else:
            nnc_mdl, nnc_mdl_executer, model_parameters = tensorflow_model.create_NNC_model_instance_from_object(
                model_path_or_object,
                dataset_path=dataset_path,
                lr=learning_rate,
                batch_size=batch_size,
                num_workers=num_workers,
                model_struct=model_struct,
                model_name=model_name
                )
    
    # Check if the model is a PyTorch model 
    elif pytorch_model.is_pyt_model(model_path_or_object):
        is_pyt_model = True
        
        # Create an NNC model instance from the PyTorch model object or path
        if model_executer:
                nnc_mdl, _, model_parameters = pytorch_model.create_NNC_model_instance_from_object(
                 model_path_or_object,
                )
        else:    
            # CURRENT IMPLEMENTED NERF-LSA ONLY USE THIS TYPE OF NNC-MODEL INITIALIZATION
            nnc_mdl, nnc_mdl_executer, model_parameters = pytorch_model.create_NNC_model_instance_from_object(
                model_path_or_object,
                dataset_path=dataset_path,
                lr=learning_rate,
                batch_size=batch_size,
                num_workers=num_workers,
                model_struct=model_struct,
                lsa=lsa,
                epochs=epochs,
                max_batches=max_batches,
                task_type = task_type,
                dataset_type = dataset_type,
                N_iters = N_iters, 
                learning_rate_decay = learning_rate_decay, 
                i_save = i_save
                )
            
    
    elif os.path.exists( os.path.expanduser(model_path_or_object)):
        model_path_or_object = os.path.expanduser(model_path_or_object)
        
        # Check if the model is a TensorFlow model
        if model_path_or_object.endswith(".h5") or model_path_or_object.endswith(".hdf5") or model_path_or_object.endswith(".tf"):
            if bnf:
                print("WARNING: Batch-norm folding (BNF) requires the tensors to be shaped such that the first dimensions corresponds to the number of output channels, which is usually not the case for TensorFlow. For further details refer to the Wiki!")
            if lsa:
                print("INFO: LSA not yet supported for TensorFlow models. 'lsa' has been set to false!")
                lsa = False
            is_tef_model = True
            
            if model_executer:
                nnc_mdl, _, model_parameters = tensorflow_model.create_NNC_model_instance_from_file(
                 model_path_or_object,
                )
                
            else:
                nnc_mdl, nnc_mdl_executer, model_parameters = tensorflow_model.create_NNC_model_instance_from_file(
                    model_path_or_object,
                    dataset_path=dataset_path,
                    lr=learning_rate,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    model_struct=model_struct,
                    model_name=model_name
                    )
                
        # Check if the model is a PyTorch model 
        elif model_path_or_object.endswith(".pt") or model_path_or_object.endswith(".pth"):
            is_pyt_model = True
            
            # Create an NNC model instance from the PyTorch model file
            if model_executer:
                nnc_mdl, _, model_parameters = pytorch_model.create_NNC_model_instance_from_file(
                 model_path_or_object,
                )
            else:    
                nnc_mdl, nnc_mdl_executer, model_parameters = pytorch_model.create_NNC_model_instance_from_file(
                    model_path_or_object,
                    dataset_path=dataset_path,
                    lr=learning_rate,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    model_struct=model_struct,
                    lsa=lsa,
                    epochs=epochs,
                    max_batches=max_batches,
                    )

        else:
            nnc_mdl, model_parameters = nnr_model.create_NNC_model_instance_from_file( model_path_or_object )
            nnc_mdl_executer = None

    else:
        raise SystemExit("Can't find path or object {}".format(model_path_or_object))

    if model_executer:
        nnc_mdl_executer = model_executer    
    
    '''
    This section checks if block_id_and_param_type is None and if BNF or LSA is applicable for the model. 
    If these conditions are satisfied, it guesses the block_id_and_param_type from the model_parameters and performs a sanity check. 
    If the sanity check fails, it resets block_id_and_param_type to None, sets the lsa and bnf flags to False, 
    and performs some additional cleanup related to the model_executer and model_parameters.
    '''
    if block_id_and_param_type is None and (bnf or lsa) and (is_pyt_model or is_tef_model):
        """
        block_id_and_param_type:
    
            {key name of state_dict : custom name used in this NNCodec}
            
            ex) {'model.rgb_linear.weight' : 'weight' ...}
        """
        # Guess the block_id_and_param_type from the model parameters
        block_id_and_param_type = nnc_mdl.guess_block_id_and_param_type(model_parameters)
        
        # Perform a sanity check on block_id_and_param_type       
        blkIdParamTypeOk = nnc_core.nnr_model.sanity_check_block_id_and_param_type( block_id_and_param_type, model_parameters )
        
        # If the sanity check fails, reset block_id_and_param_type and flags
        if blkIdParamTypeOk == False:
            print("INFO: Sanity check for block_id_and_param_type failed! block_id_and_param_type has been set to 'None', and the flags 'lsa' and 'bnf' have been set to 'False'!")
            block_id_and_param_type = None
            lsa = False
            bnf = False
            
            # Reset the model_executer to use the original model
            if model_executer:
                model_executer.model = model_executer.original_model
                del model_executer.original_model
            
            # Remove weight_scaling related keys from model_parameters
            for key in model_parameters.keys():
                if "weight_scaling" in key:
                    del model_parameters[key]
            

    bitstream = compress(model_parameters,
                         bitstream_path=bitstream_path,
                         qp=qp,
                         qp_density=qp_density,
                         nonweight_qp=nonweight_qp,
                         qp_per_tensor=qp_per_tensor,
                         use_dq=use_dq,
                         codebook_mode=codebook_mode,
                         scan_order=scan_order,
                         lambda_scale=lambda_scale,
                         param_opt=param_opt,
                         cabac_unary_length_minus1=cabac_unary_length_minus1,
                         opt_qp=opt_qp,
                         ioq=ioq,
                         bnf=bnf,
                         lsa=lsa,
                         fine_tune=fine_tune,
                         block_id_and_param_type=block_id_and_param_type,
                         model=nnc_mdl,
                         model_executer=nnc_mdl_executer,
                         verbose=verbose,
                         return_bitstream=return_bitstream,
                         )
    
    if return_model_data==True:
        if return_bitstream:
            return bitstream, block_id_and_param_type
        else:
            return block_id_and_param_type
    elif return_bitstream:
        return bitstream


def compress( 
    parameter_dict,
    bitstream_path="./bitstream.nnc",
    qp=-38,
    qp_density=2,
    nonweight_qp=-75,
    qp_per_tensor=None,
    use_dq=True,
    codebook_mode=0,
    scan_order=0,
    lambda_scale=0,
    param_opt=True,
    cabac_unary_length_minus1=10,
    opt_qp=False,
    ioq=False,
    bnf=False,
    lsa=False,
    fine_tune=False,
    block_id_and_param_type=None,
    model=None,
    model_executer=None,
    verbose=True,
    return_bitstream=False,
    ):

    """
    Compresses a model's parameters into a bitstream.

    Args:
    - parameter_dict (dict): 
        same as model_parameter of compress_model, and model_data of init_model_from_dict. e.g.
        model_data = {
            'parameters': {
                'fc1.weight': array([[ 0.1091,  0.0297, -0.0385, -0.0356, -0.0826,  0.2660],
                                    [-0.0881, -0.0847, -0.1659,  0.0734,  0.1251, -0.1607],
                                    [-0.0431, -0.0939, -0.1145,  0.2592,  0.1930, -0.1175]], dtype=float32),
                'fc1.bias': array([-0.2314,  0.1364, -0.1852, -0.0803,  0.0837], dtype=float32),
                'fc2.weight': array([[ 0.0561, -0.1871,  0.0493, -0.0184,  0.0457],
                                    [-0.0767, -0.2360, -0.2071, -0.0615, -0.2126]], dtype=float32),
                'fc2.bias': array([-0.1852, -0.2391], dtype=float32)
                },
            'reduction_method': 'baseline'
            }
        ...

    Returns:
    - bitstream (bytearray)
        The compressed model as a bitstream (if return_bitstream is True).
    """
    
    ''' SUMMARY
    1. It checks the validity of the parameter dictionary and creates an NNRModel instance based on the model or parameter dictionary.
    2. If a block_id_and_param_type is provided, it performs a sanity check and sets the block_id_and_param_type in the model.
    3. If a model_executer is available, it checks its capabilities for LSA, FT, and IOQ.
    4. The function initializes the approximator data and creates an ApproxInfo object based on the parameters.
    5. If IOQ (Inference-based Quantization) is enabled, it optimizes the quantization parameters using the model_executer.
    6. If LSA (Local Scale Adaptation) or FT (Fine-tuning) is enabled, it runs the respective preprocessing steps using the model_executer.
    7. If BNF (BatchNorm Folding) is enabled, it performs folding of batch normalization layers.
    8. It performs approximation using the approximator, which quantizes the model parameters based on the specified method and parameters.
    9. It encodes the approximated data using the coder, generating a compressed bitstream.
    10.The original size of the model and the compressed size are printed, along with the compression ratio and execution time.
    11. If a bitstream path is provided, the bitstream is saved to the specified file.
    12. If return_bitstream is True, the compressed bitstream is returned.
    '''
    
    # Initialization of the approximator and encoder
    try:
        start = timer()
        start_overall = start
        
        # Print initialization message
        __print_output_line("INITIALIZE APPROXIMATOR AND ENCODER...", verbose=verbose)
        
        # Checks the input parameter_dict and initializes the model and model_executer        
        if isinstance(parameter_dict, dict) and all( [isinstance(a, np.ndarray) for a in parameter_dict.values()] ) and (all([ (a.dtype==np.float32 or a.dtype==np.int32) for a in parameter_dict.values()])):
            model_parameters = parameter_dict
            
            # If model is already an NNRModel instance, use it directly            
            if isinstance(model, nnc_core.nnr_model.NNRModel):
                nnc_mdl = model
            else:
                # If not NNRModel instance, Create an NNRModel instance from the parameter_dict
                nnc_mdl = nnc_core.nnr_model.NNRModel(parameter_dict)

            if model_executer is not None:
                assert isinstance( model_executer, nnc_core.nnr_model.ModelExecute ), "model_executer must be of type ModelExecute!"
        else:
            raise SystemExit("Parameter dict must be a dict (key-value pairs). The keys shall be stings, specifying the tensor names. The values shalls be numpy arrays (ndarray) of type float32 or int32!")
    except:
        raise SystemExit("Can not read parameter_dict: {}".format(parameter_dict))

    # Set the block_id_and_param_type if provided
    if block_id_and_param_type is not None:
        blkIdParamTypeOk = nnc_core.nnr_model.sanity_check_block_id_and_param_type( block_id_and_param_type, parameter_dict )
        
        if blkIdParamTypeOk:
            nnc_core.nnr_model.set_block_id_and_param_type( nnc_mdl.model_info , block_id_and_param_type )
            
        else:
            print("INFO: Sanity check for block_id_and_param_type failed!"
                  "block_id_and_param_type has been set to 'None',"
                  "and the flags 'lsa' and 'bnf' have been set to 'False'!")
            block_id_and_param_type = None
            lsa = False
            bnf = False
            
    if model_executer:
        if lsa and not model_executer.has_tune_lsa():
            print("INFO: Tuning (training) of LSA parameters (tune_model) not implemented by model_executer! "
                  "'lsa' has been set to 'False'!")
            lsa = False
        if fine_tune and not model_executer.has_tune_ft():
            print("INFO: Fine tuning (training) of parameters (tune_model) not implemented by model_executer! "
                  "'fine_tune' has been set to 'False'!")
            fine_tune = False
        if ioq and not model_executer.has_eval():
            print("INFO: Evaluation (inference on a reduced dataset) of parameters (eval_model) not implemented by model_executer! "
                  "ioq' has been set to 'False'!")
            ioq = False
                    
                    
    # INITIALIZATION
    # Initialize approx_data for the approximator
    approx_data =  nnc_core.approximator.init_approx_data(  model_parameters,
                                                            nnc_mdl.model_info, 
                                                            qp_density=qp_density, 
                                                            scan_order=scan_order
                                                         )
    
    # Create ApproxInfo object for the approximator
    ApproxInfoO = nnc_core.approximator.ApproxInfo( approx_data,
                                                    nnc_mdl.model_info,
                                                    "uniform" if codebook_mode==0 else "codebook",
                                                    codebook_mode,
                                                    qp,
                                                    opt_qp,
                                                    not use_dq,
                                                    cabac_unary_length_minus1,
                                                    lambda_scale,
                                                    nonweight_qp=nonweight_qp,
                                                    qp_per_tensor=qp_per_tensor
                                                )
    approx_info = ApproxInfoO.approx_info


    # Set encoding information
    enc_info = {
            "cabac_unary_length_minus1" : cabac_unary_length_minus1,
            "param_opt_flag"     : param_opt,
        }
    end = timer()
    
    __print_output_line("DONE in {:.4f} s\n".format(end-start), verbose=verbose)

    #PREPROCESSING
    if ioq:
        assert model_executer is not None, "model_executer must be available in order to run IOQ!"
        
        start = timer()
        __print_output_line("PREPROCESSING, IOQ...\n", verbose=verbose) 
        
        nnc_core.approximator.inference_based_qp_opt(
            approx_info, 
            nnc_mdl.model_info,
            model_executer,
            approx_data,
            enc_info["param_opt_flag"],
            enc_info["cabac_unary_length_minus1"],
            verbose=verbose,
        )
        end = timer()
        __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)   

    # LSA and FT
    # Run LSA and/or FT if enabled
    if lsa or fine_tune:
        
        assert model_executer is not None, "model_executer must be available in order to run LSA and/or FT!"
        
        start = timer()
        
        __print_output_line("PREPROCESSING, LSA/FT...\n", verbose=verbose) 
        
        nnc_core.approximator.run_ft_and_lsa(
            nnc_mdl.model_info,
            approx_data,
            ApproxInfoO,
            model_executer,
            block_id_and_param_type,
            lsa,
            fine_tune, 
            use_dq,
            verbose,
            bitstream_path
        )
        end = timer()
        
        __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)  
        
    # BNF
    # Perform BatchNorm Folding if enabled
    if bnf:
        start = timer()
        
        __print_output_line("PREPROCESSING, BNF...", verbose=verbose)    
        nnc_core.approximator.fold_bn(nnc_mdl.model_info, approx_data, ApproxInfoO)
        
        end = timer()
        
        __print_output_line("DONE in {:.4f} s\n".format(end-start), verbose=verbose)

    # QUANTIZATION AND ENCODING
    start = timer() 
    __print_output_line("APPROXIMATING WITH METHOD {}...".format(approx_info["approx_method"]), verbose=verbose)
    
    # Perform approximation using the approximator
    approx_data_enc = nnc_core.approximator.approx( approx_info,
                                                nnc_mdl.model_info,
                                                approx_data,
                                                enc_info["param_opt_flag"]
                                               )
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)

    start = timer()
    __print_output_line("ENCODING...", verbose=verbose)
    
    # Encode the approximated data using the coder
    bitstream = nnc_core.coder.encode(  enc_info, 
                                    nnc_mdl.model_info, 
                                    approx_data_enc
                                 )
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)

    original_size = nnc_mdl.model_info["original_size"]

    __print_output_line("COMPRESSED FROM {} BYTES TO {} BYTES ({} KB, {} MB, COMPRESSION RATIO: {:.2f} %) in {:.4f} s\n".format(original_size, len(bitstream), len(bitstream)/1000.0, len(bitstream)/1000000.0, len(bitstream)/original_size*100, end-start_overall), verbose=verbose)
    
    if bitstream_path is not None:
        # Save the bitstream to the specified path
        with open( bitstream_path, "wb" ) as br_file:
            br_file.write( bitstream )

    if return_bitstream:
        return bitstream


def decompress( bitstream_or_path, 
                block_id_and_param_type=None, 
                return_model_information=False, 
                verbose=True, 
                reconstruct_lsa=True, 
                reconstruct_bnf=True
                ):

    """
    Decompresses a model's bitstream into parameters.

    Args:
    - bitstream_or_path (bytearray or str): The bitstream or path to the bitstream of the compressed model.
    - block_id_and_param_type (dict): Optional dictionary of block identifiers and parameter types.
    - return_model_information (bool): Whether to return model information after decompression.
    - verbose (bool): Whether to print verbose output.
    - reconstruct_lsa (bool): Whether to reconstruct Layer Scale Approximation (LSA).
    - reconstruct_bnf (bool): Whether to reconstruct Batch Normalization Folding (BNF).

    Returns:
    - parameters (dict): The decompressed model parameters.
    - model_information (dict): Additional model information (if return_model_information is True).
    """

    # Initialize a dictionary to hold the decompressed model's information
    dec_model_info  = {'parameter_type': {},
                      'parameter_dimensions': {},
                      'parameter_index': {},
                      'block_identifier': {},
                      'topology_storage_format' : None,
                      'topology_compression_format' : None,
                      'performance_maps' : { "mps" : {}, "lps" : {}},
                      'performance_map_flags' : { "mps_sparsification_flag" : {}, "lps_sparsification_flag" : {},
                                                  "mps_pruning_flag" : {}, "lps_pruning_flag" : {},
                                                  "mps_unification_flag" : {}, "lps_unification_flag" : {},
                                                  "mps_decomposition_performance_map_flag" : {}, "lps_decomposition_performance_map_flag" : {},
                                                } 
                      }

    model_information = { 'topology_storage_format' : None,
                          'performance_maps' : {},
                          'performance_map_flags' : {}
                        }

    # If a block_id_and_param_type is provided, perform a sanity check
    if block_id_and_param_type is not None:
        blkIdParamTypeOk = nnc_core.nnr_model.sanity_check_block_id_and_param_type( block_id_and_param_type )
        
        if blkIdParamTypeOk == False:
            print("INFO: Sanity check for block_id_and_param_type failed! block_id_and_param_type has been set to 'None'!")
            block_id_and_param_type = None
        else:
            nnc_core.nnr_model.set_block_id_and_param_type( dec_model_info, block_id_and_param_type )

    hls_bytes = {}
    start = timer()
    __print_output_line("DECODING...", verbose=verbose)

    # Set up the bitstream
    if isinstance(bitstream_or_path, bytearray):
        bitstream = bitstream_or_path
        
    elif os.path.exists(os.path.expanduser(bitstream_or_path)):
        with open( os.path.expanduser(bitstream_or_path), "rb" ) as br_file:
            bitstream = br_file.read()
            
    else:
        raise SystemExit( "Could not read bitstream or bitstream_path: {}".format(bitstream_or_path) )

    # Call the decode function from the coder module to decode the bitstream
    dec_approx_data = nnc_core.coder.decode(bitstream, dec_model_info, hls_bytes)
    
    end = timer()
    
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)

    start = timer()
    
    rec_approx_data = dec_approx_data
    __print_output_line("RECONSTRUCTING...", verbose=verbose)
    
    # Call the rec function from the approximator module to reconstruct the model parameters
    nnc_core.approximator.rec(rec_approx_data )
    
    # If the reconstruct_bnf flag is set, unfold the batch normalization layers
    if reconstruct_bnf:
        nnc_core.approximator.unfold_bn(dec_model_info, rec_approx_data)
        
    # If the reconstruct_lsa flag is set, apply the layer scale approximation
    if reconstruct_lsa:
        nnc_core.approximator.apply_lsa(dec_model_info, rec_approx_data)
        
    # Recompose the parameters into the original model form
    rec_approx_data = nnc_core.approximator.recompose_params( dec_model_info, rec_approx_data)
    end = timer()
    
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)
    
    # Return the reconstructed parameters and optionally the model information
    if return_model_information:
        model_information["topology_storage_format"] = dec_model_info["topology_storage_format"]
        model_information["performance_maps"]        = dec_model_info["performance_maps"]
        model_information["performance_map_flags"]   = dec_model_info["performance_map_flags"]

        return rec_approx_data["parameters"], model_information
    else:
        return rec_approx_data["parameters"]


def decompress_model( bitstream_or_path,
                      model_path=None,#"./rec.mdl",
                      block_id_and_param_type=None,
                      model_struct=None,
                      model_executer=None,
                      model_name=None, 
                      dataset_path=None, 
                      batch_size=64, 
                      num_workers=8,
                      reconstruct_bnf=True,
                      reconstruct_lsa=True,
                      test_model=False,
                      return_model_information=False,
                      return_decompressed_model=False,
                      verbose=True,
                    ):
    """
    Decompresses a model's bitstream into model parameters and optionally tests the model's performance.

    Args:
    - bitstream_or_path (bytearray or str): The bitstream or path to the bitstream of the compressed model.
    - model_path (str): Path to save the decompressed model.
    - block_id_and_param_type (dict): Optional dictionary of block identifiers and parameter types.
    - model_struct (ModelStruct): The structure of the model to be decompressed
    - model_executer (ModelExecute): The ModelExecute instance to execute the model.
    - model_name (str): The name of the model.
    - dataset_path (str): The path to the dataset for testing the model.
    - batch_size (int): The batch size for testing the model.
    - num_workers (int): The number of worker threads for data loading.
    - reconstruct_bnf (bool): Whether to reconstruct Batch Normalization Folding (BNF).
    - reconstruct_lsa (bool): Whether to reconstruct Layer Scale Approximation (LSA).
    - test_model (bool): Whether to test the model after decompression.
    - return_model_information (bool): Whether to return model information after decompression.
    - return_decompressed_model (bool): Whether to return the decompressed model.
    - verbose (bool): Whether to print verbose output.

    Returns:
    - parameters (dict): The decompressed model parameters.
    - model_information (dict): Additional model information (if return_model_information is True).
    - model_with_decoded_parameters: The model with decoded parameters (if return_decompressed_model is True).
    """
    
    
    
    # If a block_id_and_param_type is provided, perform a sanity check
    if block_id_and_param_type is not None:
        blkIdParamTypeOk = nnc_core.nnr_model.sanity_check_block_id_and_param_type( block_id_and_param_type )
        
        if blkIdParamTypeOk == False:
            print("INFO: Sanity check for block_id_and_param_type failed! block_id_and_param_type has been set to 'None'!")
            block_id_and_param_type = None

    # Call the decompress function to decompress the model
    model_dict, model_information = decompress(bitstream_or_path, 
                                        block_id_and_param_type=block_id_and_param_type, 
                                        return_model_information=True,
                                        reconstruct_lsa=reconstruct_lsa,
                                        reconstruct_bnf=reconstruct_bnf
                                       )

    model_with_decoded_parameters = None
    
    # Handle the decompressed model based on its topology storage format
    if model_information["topology_storage_format"] == nnc_core.nnr_model.TopologyStorageFormat.NNR_TPL_PYT:
        
        # Default path for PyTorch model if not provided
        if model_path == None:
            model_path="./rec.pt"

        # Save decompressed model in PyTorch format
        pytorch_model.save_to_pytorch_file( model_dict, model_path )
        
        # If model structure and dataset are available, or model executer is provided, and test_model flag is set, test the model
        if ( (model_struct and dataset_path) or model_executer ) and test_model:
            
            # If model executer is provided, use it, otherwise create a new one
            if model_executer:
                nnc_mdl_executer = model_executer
            else:    
                _, nnc_mdl_executer, _ = pytorch_model.create_NNC_model_instance_from_file(
                    model_path,
                    dataset_path=dataset_path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    model_struct=model_struct,
                    )

            # Test the model and print the accuracy
            acc = nnc_mdl_executer.test_model(
                    model_dict,
                    verbose=verbose
                    )
            print(acc)
        
        # If model structure is available and return_decompressed_model flag is set, return the model with decoded parameters
        if model_struct and return_decompressed_model:
            model_with_decoded_parameters = pytorch_model.get_model_file_with_parameters(model_struct=model_struct, parameters=model_dict)
                
    elif model_information["topology_storage_format"] == nnc_core.nnr_model.TopologyStorageFormat.NNR_TPL_TEF:

        # Default path for TensorFlow model if not provided
        if model_path == None:
            model_path="./rec.h5"
            
        # Save decompressed model in TensorFlow format
        tensorflow_model.save_to_tensorflow_file( model_dict, model_path )
        
        # If model structure and dataset are available, or model executer is provided, and test_model flag is set, test the model
        if ( (model_struct and dataset_path) or model_executer ) and test_model:
            
            # If model executer is provided, use it, otherwise create a new one
            if model_executer:
                nnc_mdl_executer = model_executer
                
            else:
                _, nnc_mdl_executer, _ = tensorflow_model.create_NNC_model_instance_from_file(
                    model_path,
                    dataset_path=dataset_path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    model_struct=model_struct,
                    model_name=model_name 
                    )
                
            # Test the model and print the accuracy
            acc = nnc_mdl_executer.test_model(
                    model_dict,
                    verbose=verbose
                    )
            print(acc)
            
        # If model structure is available and return_decompressed_model flag is set, return the model with decoded parameters
        if model_struct and return_decompressed_model:
           model_with_decoded_parameters = tensorflow_model.get_model_file_with_parameters(model_struct=model_struct, parameters=model_dict)

    elif model_information["topology_storage_format"] == nnc_core.nnr_model.TopologyStorageFormat.NNR_TPL_UNREC or model_information["topology_storage_format"] == None:
        
        # Default path for unrecognized topology storage format if not provided
        if model_path == None:
            model_path="./rec.mdl"

        # Save decompressed model in pickled format
        nnr_model.save_to_pickled_file( model_dict, model_path )

        # If model executer is provided and test_model flag is set, test the model
        if model_executer and test_model:
            nnc_mdl_executer = model_executer
            
            # Test the model and print the accuracy
            acc = nnc_mdl_executer.test_model(
                model_dict,
                verbose=verbose
            )
            print(acc)

    else:
        # Raise error if topology storage format is not recognized
        raise SystemExit( "Topology Storage Format not yet supported: {}".format( model_information["topology_storage_format"] ) )

    # Return the reconstructed parameters and optionally the model information
    if return_decompressed_model and return_model_information:
        return model_with_decoded_parameters, model_information
    
    elif return_decompressed_model:
        return model_with_decoded_parameters
    
    elif return_model_information:
        return model_information