import copy
import numpy as np
import deepCABAC
import nnc_core
from nnc_core.nnr_model import NNRModelAccess
from nnc_core.coder import hls, baseline
from .. import common


def approx(approx_info, model_info, approx_data_in):
    """
    Approximates the parameters of a model.

    Args:
    - approx_info (dict): Information about the approximation, including parameters to approximate, quantization parameter (qp), dq_flag, lambda_scale, and cabac_unary_length_minus1.
    - model_info (dict): Information about the model.
    - approx_data_in (dict): The input data for approximation, including parameters, compressed_parameter_types, qp_density, and scan_order.

    Returns:
    - approx_data_out (dict): The output data after approximation, including parameters, approx_method, qp, and dq_flag.
    """
    
    approx_data_out = {k: copy.copy(v) for k, v in approx_data_in.items()} # create copies of dicts in approx_data
    encoder = deepCABAC.Encoder() # Initialize the deepCABAC encoder
    model_access = NNRModelAccess(model_info) # Initialize the NNRModelAccess for the model
    
    # Loop over the blocks and parameters of the model
    for block_or_param in model_access.blocks_and_params():
        for par_type, param, _ in block_or_param.param_generator(approx_data_in["compressed_parameter_types"]):
            
            # Check if the parameter type is in the to_approximate list and the parameter is not in the approx_method dict
            if (par_type in approx_info["to_approximate"]) and (param not in approx_data_in["approx_method"]):
                
                # !!! There seems to be a pybind11 issue when using np.zeros_like for "values" that have been transposed.
                # !!! It seems that sometimes, encoder.quantLayer returns only zeros for quantizedValues. Needs further study.
                # !!! For now, using np.zeros instead of np.zeros_like seems to be a workaround.
                
                # Create a zero array with the shape of the parameter
                quantizedValues = np.zeros(approx_data_in["parameters"][param].shape, dtype=np.int32)
                
                # Initialize the context models of the encoder
                encoder.initCtxModels( approx_info["cabac_unary_length_minus1"], 0 )

                # Get the quantization parameter for the current parameter
                enc_qp = approx_info['qp'][param]

                # Quantize the layer
                qp = encoder.quantLayer(
                    approx_data_in["parameters"][param],
                    quantizedValues,
                    approx_info['dq_flag'][param],
                    approx_data_out['qp_density'],
                    enc_qp,
                    approx_info["lambda_scale"],
                    approx_info["cabac_unary_length_minus1"],
                    approx_data_in["scan_order"].get(param, 0)
                )

                # Check if the quantization parameter has been clipped
                if qp != enc_qp:
                    print("INFO: QP for {} has been clipped from {} to {} to avoid int32_t overflow!".format(param, approx_info['qp'][param],qp))
                    approx_data_out['qp'][param] = qp
                else:
                    approx_data_out['qp'][param] = enc_qp

                # Update the approx_data_out dict
                approx_data_out['parameters'][param] = quantizedValues
                approx_data_out['approx_method'][param] = 'uniform'
                approx_data_out['dq_flag'][param] = approx_info['dq_flag'][param]
   
    return approx_data_out

def rec(param, approx_data):
    """
    Reconstructs a parameter from its approximation.

    Args:
    - param (str): The name of the parameter to reconstruct.
    - approx_data (dict): The data for reconstruction, including parameters, qp_density, qp, and scan_order.

    Returns:
    - None. The approx_data dict is updated in place.
    """
    
    # Check that the parameters are of type int32
    assert approx_data['parameters'][param].dtype == np.int32
    
    # Initialize the deepCABAC decoder
    decoder = deepCABAC.Decoder()

    # Get the values for the parameter
    values = approx_data['parameters'][param]

    # Create a zero array with the shape of the values
    approx_data["parameters"][param] = np.zeros(values.shape, dtype=np.float32)
    
    # Dequantize the layer
    decoder.dequantLayer(approx_data["parameters"][param], values, approx_data["qp_density"], approx_data["qp"][param], approx_data['scan_order'].get(param, 0))

    # Remove the parameter from the approx_method dict
    del approx_data["approx_method"][param]