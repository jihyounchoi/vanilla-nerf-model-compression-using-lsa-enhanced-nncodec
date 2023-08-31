
# This module provides functionality to skip approximation for parameters that are already of integer data type.
# The parameters that are skipped will not be further approximated or modified. 
# The 'skip' approximation method is essentially a no-op: it doesn't change the parameter values at all. 
# This can be useful for preserving parameters that are already in a desirable format or for testing and debugging purposes.

import copy
import numpy as np
from nnc_core.nnr_model import NNRModelAccess

def skip_approx(approx_info, model_info, approx_data_in):
    """
    This function skips the approximation process for parameters that are already of integer data type.

    Args:
        approx_info (dict): A dictionary containing information about the approximation process.
        model_info (dict): A dictionary containing information about the model.
        approx_data_in (dict): A dictionary containing the data to be approximated.

    Returns:
        approx_data_out (dict): A dictionary containing the approximated data.
    """
    
    # create a copy of the input data dictionary.
    approx_data_out = {k: copy.copy(v) for k, v in approx_data_in.items()}
    
    # create an instance of NNRModelAccess with the model information.
    model_access = NNRModelAccess(model_info)
    
    # loop over blocks and parameters of the model.
    for block_or_param in model_access.blocks_and_params():
        for par_type, param, _ in block_or_param.param_generator(approx_data_in["compressed_parameter_types"]):
            
            # if the parameter type is in the list to be approximated, and it has not been approximated yet, and the parameter data type is integer
            if (par_type in approx_info["to_approximate"]) and (param not in approx_data_in["approx_method"]) and ( approx_data_in["parameters"][param].dtype == np.int32 ):
            
                # set the approximation method for this parameter to 'skip' and the dq_flag to 0.
                approx_data_out['approx_method'][param] = 'skip'
                approx_data_out['dq_flag'][param] = 0
   
    # return the modified approximation data dictionary.
    return approx_data_out


def skip_rec(param, approx_data):
    """
    This function deletes the approximation method for the given parameter from the data.

    Args:
        param (str): The parameter for which the approximation method is to be deleted.
        approx_data (dict): A dictionary containing the approximated data.

    Returns:
        None
    """
    
    # assert that the data type of the parameter is integer.
    assert approx_data['parameters'][param].dtype == np.int32

    # delete the approximation method for this parameter from the data.
    del approx_data["approx_method"][param]

