import numpy as np
import sys
import copy
import multiprocessing
import time
import traceback
from .. import common
import deepCABAC
from nnc_core.coder import hls
import nnc_core
from nnc_core.nnr_model import NNRModelAccess


def derive_sorted_codebook_from_tensor(tensor):
    """
    Derives a sorted codebook and corresponding indices from a tensor.

    Args:
    tensor : np.ndarray
        The input tensor.

    Returns:
    codebook : np.ndarray
        The sorted unique elements of the tensor.
    reshaped_indices : np.ndarray
        The indices of the sorted unique elements in the original tensor.
    """
    
    # Get the shape of the original tensor
    originalShape = tensor.shape 
    
    # Create a codebook and indices from unique elements of the tensor
    codebook, indices = np.unique(tensor, return_inverse=True)
    
    # Reshape the indices to the original shape
    reshaped_indices = indices.reshape( originalShape )
    
    # Return the codebook and reshaped indices as int32
    return codebook, reshaped_indices.astype('int32')

def get_codebook_offset( codebook, indices, cabac_unary_length_minus1 ):
    """
    Returns the codebook offset.

    Args:
    codebook : np.ndarray
        The codebook.
    indices : np.ndarray
        The indices of the codebook.
    cabac_unary_length_minus1 : int
        The length of the unary code minus 1.

    Returns:
    codebook : np.ndarray
        The updated codebook.
    indexes : np.ndarray
        The updated indices.
    codebookOffset : int
        The codebook offset.
    """
    
    # Initialize the codebook offset to 0
    codebookOffset = 0
    
    # Check if the dtype of indices is int32
    if indices.dtype == np.int32:
        codebookOffset = -1
        minBits = None
        
        # Iterate over the range of the length of the codebook
        for cb in range( len( codebook ) ):
            encoder = deepCABAC.Encoder() # Initialize the encoder
            
            # Initialize context models in the encoder
            encoder.initCtxModels( cabac_unary_length_minus1, 1 )
            
            # Subtract the current index from the indices
            indexes = indices - cb
            
            # Encode the layer with the updated indices
            encoder.encodeLayer( indexes, 0, 0 )
            
            # Get the number of bits in the encoded data
            bits = len( encoder.finish().tobytes() )

            # Update the minimum bits and codebook offset if necessary
            if minBits == None or bits < minBits:
                minBits = bits
                codebookOffset = cb
                
    # Update the indices by subtracting the codebook offset
    indexes = indices - codebookOffset
    
    # Return the codebook, updated indices, and codebook offset
    return codebook, indexes, codebookOffset

def get_best_egk(codebook, codebookOffset):
    """
    Returns the best EGK (Exponential Golomb code of order k) for the given codebook and codebook offset.

    Args:
    codebook : np.ndarray
        The codebook.
    codebookOffset : int
        The codebook offset.

    Returns:
    best_egk : int
        The best EGK.
    min_bytes_cb : int
        The minimum number of bytes for the codebook.
    """
    
    # Initialize a dictionary for the HLS codebook
    cb_hls = {}
    cb_hls["CbZeroOffset__"] = codebookOffset
    cb_hls["codebook_size__"] = len( codebook )
    cb_hls["codebook__"] = codebook

    min_bytes_cb = None
    
    # Iterate over the range of 16 to find the best egk
    for i in range(16):
        cb_hls["codebook_egk__"] = i
        bs = bytearray()
        w = hls.BitWriter(bs)
        hls_enc = hls.Coder( w, cb_hls )
        hls_enc.codebook("")
        bytes_cb = w.getNumBitsTouched()
        
        # Update the minimum bytes and best egk if necessary
        if min_bytes_cb is None or bytes_cb < min_bytes_cb:
            min_bytes_cb = bytes_cb
            best_egk = i
            
    # Return the best egk and minimum bytes
    return best_egk, min_bytes_cb

def get_codebook_bytes(codebook, codebookOffset, cbEgk):
    """
    Returns the number of bytes in the codebook.

    Args:
    codebook : np.ndarray
        The codebook.
    codebookOffset : int
        The codebook offset.
    cbEgk : int
        The EGK of the codebook.

    Returns:
    bytes_cb : int
        The number of bytes in the codebook.
    """
    
    # Initialize a dictionary for the HLS codebook
    cb_hls = {}
    cb_hls["CbZeroOffset__"] = codebookOffset
    cb_hls["codebook_size__"] = len( codebook )
    cb_hls["codebook__"] = codebook
    cb_hls["codebook_egk__"] = cbEgk
    
    bs = bytearray()
    w = hls.BitWriter(bs)
    hls_enc = hls.Coder( w, cb_hls )
    hls_enc.codebook("")
    bytes_cb = w.getNumBitsTouched()
   
    # Return the number of bytes touched
    return bytes_cb

def approx(approx_info, model_info, approx_data_in, param_opt=0):
    """
    Approximates the tensor using uniform quantization without DQ (Dependent Quantization).

    Args:
    approx_info : dict
        Information about the approximation.
    model_info : dict
        Information about the model.
    approx_data_in : dict
        The input data for approximation.
    param_opt : int, optional
        The parameter option for approximation.

    Returns:
    approx_data_out : dict
        The output data after approximation.
    approx_info_out : dict
        The output information after approximation.
    """
    ##Qunatize tensor with uniform but without DQ

    # Make a copy of the input data
    approx_data_out = {k: copy.copy(v) for k, v in approx_data_in.items()} # create copies of dicts in approx_data
    
    # Initialize the encoder
    encoder = deepCABAC.Encoder()

    # Create a model access object
    model_access = NNRModelAccess(model_info)
    
    # Iterate over the blocks and parameters in the model
    for block_or_param in model_access.blocks_and_params():
        for par_type, param, _ in block_or_param.param_generator(approx_data_in["compressed_parameter_types"]):
            
            # Check if the parameter type needs to be approximated and hasn't been approximated already
            if (par_type in approx_info["to_approximate"]) and (param not in approx_data_in["approx_method"]):
                # !!! There seems to be a pybind11 issue when using np.zeros_like for "values" that have been transposed.
                # !!! It seems that sometimes, encoder.quantLayer returns only zeros for quantizedValues. Needs further study.
                # !!! For now, using np.zeros instead of np.zeros_like seems to be a workaround.
                
                # Initialize the quantizedValues array
                quantizedValues = np.zeros(approx_data_in["parameters"][param].shape, dtype=np.int32)
                
                # Initialize context models in the encoder
                encoder.initCtxModels( approx_info["cabac_unary_length_minus1"], 0 )
                
                # If dependent quantization is used, compute the QP offset
                qp_off = 0
                
                if approx_info['dq_flag'][param] == 1:
                    qp_off = common.compute_qp_offset_to_dq_equivalent( approx_data_out['qp_density'] )
                    print("INFO: Dependent quatization (DQ) can not be used with 'codebook'. In order to get similiar performance (to DQ) the QP is changed by {}!".format(-qp_off))

                # Calculate the quantization parameter (QP)
                enc_qp = approx_info['qp'][param] - qp_off

                # Quantize the layer
                qp = encoder.quantLayer(
                    approx_data_in["parameters"][param],
                    quantizedValues,
                    0, #approx_info['dq_flag'][param],
                    approx_data_out['qp_density'],
                    enc_qp,
                    approx_info["lambda_scale"],
                    approx_info["cabac_unary_length_minus1"],
                    approx_data_in["scan_order"].get(param, 0)
                )

                # If the QP is clipped, print a message and update the QP
                if qp != enc_qp:
                    print("INFO: QP for {} has been clipped from {} to {} to avoid int32_t overflow!".format(param, approx_info['qp'][param],qp))
                    approx_data_out['qp'][param] = qp
                else:
                    approx_data_out['qp'][param] = enc_qp

                # Derive a sorted codebook from the quantized values
                codebook, indexes = derive_sorted_codebook_from_tensor(quantizedValues)
                
                # Get the codebook offset
                codebook, indexes, codebookOffset = get_codebook_offset( codebook, indexes,  approx_info["cabac_unary_length_minus1"])
                
                # Get the best EGK
                egk, _ = get_best_egk(codebook, codebookOffset)

                # If the codebook mode is 1, update the output data with the codebook information
                if approx_info["codebook_mode"] == 1:
                    approx_data_out["parameters"][param] = indexes
                    approx_data_out["codebooks"][param] = codebook
                    approx_data_out['approx_method'][param] = 'codebook'
                    approx_data_out['dq_flag'][param] = 0
                    approx_data_out["codebook_zero_offsets"][param] = codebookOffset
                    approx_data_out['codebooks_egk'][param] = egk
                
                # If the codebook mode is 2, perform additional steps to compare the cost of uniform quantization and codebook quantization
                elif approx_info["codebook_mode"] == 2:
                    if approx_info['dq_flag'][param] == 1:
                        quantizedValues = np.zeros(approx_data_in["parameters"][param].shape, dtype=np.int32)
                        encoder.initCtxModels( approx_info["cabac_unary_length_minus1"], 0 )
                        
                        enc_qp = approx_info['qp'][param]
                        ##else quantize again with DQ
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
                    

                    # Compute Cost for encoding uniform quantized parameters
                    testEnc = deepCABAC.Encoder()
                    testEnc.initCtxModels( approx_info["cabac_unary_length_minus1"], param_opt )
                    testEnc.encodeLayer(quantizedValues, approx_info['dq_flag'][param], approx_data_in["scan_order"].get(param, 0) )
                    bs_par = bytearray( testEnc.finish().tobytes() )

                    bytesUni = len(bs_par)
                    
                    # Compute cost for codebook quantized parameters + bytes for encoding the codebooks
                    testEnc = deepCABAC.Encoder()
                    testEnc.initCtxModels( approx_info["cabac_unary_length_minus1"], param_opt )
                    testEnc.encodeLayer(indexes, 0, approx_data_in["scan_order"].get(param, 0) )
                    bs_par_cb = bytearray( testEnc.finish().tobytes() )

                    bytesCb = len(bs_par_cb) + get_codebook_bytes(codebook, codebookOffset, egk)

                    # Select the cheapest method and update the output data accordingly
                    if bytesCb < bytesUni:
                        approx_data_out["parameters"][param] = indexes
                        approx_data_out["codebooks"][param] = codebook
                        approx_data_out['approx_method'][param] = 'codebook'
                        approx_data_out['dq_flag'][param] = 0
                        approx_data_out["codebook_zero_offsets"][param] = codebookOffset
                        approx_data_out['codebooks_egk'][param] = egk
                    else:
                        if approx_info['dq_flag'][param] == 1:
                            if qp != enc_qp:
                                print("INFO: QP for {} has been clipped from {} to {} to avoid int32_t overflow!".format(param, approx_info['qp'][param],qp))
                                approx_data_out['qp'][param] = qp
                            else:
                                approx_data_out['qp'][param] = enc_qp
                        approx_data_out['parameters'][param] = quantizedValues
                        approx_data_out['approx_method'][param] = 'uniform'
                        approx_data_out['dq_flag'][param] = approx_info['dq_flag'][param]

    # The approximation info remains the same
    approx_info_out = approx_info
    
    # Return the output data and the approximation info
    return approx_data_out, approx_info_out

# **********************************************************************************************************************
def rec(param, approx_data):
    """
    Recovers the original tensor from the approximated tensor.

    Args:
    param : str
        The parameter name.
    approx_data : dict
        The approximated data.

    Returns:
    None.
    """
    
    # Check that the parameters are of integer type
    assert approx_data['parameters'][param].dtype == np.int32
    
    # Get the codebook and stepsize for the parameter
    cb = approx_data['codebooks'][param]
    stepsize = common.get_stepsize_from_qp(approx_data["qp"][param], approx_data["qp_density"])
    
    # Multiply the codebook by the stepsize
    cb = cb * stepsize
    
    # Get the offset for the codebook
    offset = approx_data['codebook_zero_offsets'][param]
    
    # Update the parameters with the decoded values from the codebook
    approx_data["parameters"][param] = cb[approx_data["parameters"][param] + offset]
    
    # Remove the approximation method and codebook information for the parameter from the data
    del approx_data["approx_method"][param]
    del approx_data['codebooks'][param]
    del approx_data['codebook_zero_offsets'][param]
    del approx_data['codebooks_egk'][param]
    del approx_data['qp'][param]