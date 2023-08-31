import copy
import sys
import numpy as np
from . import baseline
from . import codebook
from . import integer
import nnc_core
from nnc_core.nnr_model import NNRModelAccess, NNRBlockAccess, W_TYPES
from nnc_core import hls
from timeit import default_timer as timer

def __print_output_line( outputString, verbose=True ):
    """
    Prints output string if verbose is True.
    
    Args:
    - outputString (str): The string to be printed.
    - verbose (bool): If True, prints the string. Defaults to True.
    """
    if verbose:
        # Print the string
        sys.stdout.write(outputString)
        sys.stdout.flush()

def del_param(approx_data, approx_info, param):
    """
    Delete a parameter from the approximation data and info.
    
    Args:
    - approx_data (dict): A dictionary containing approximation data.
    - approx_info (dict): A dictionary containing approximation info.
    - param (str): The parameter to be deleted.
    """
    # Delete the parameter from the approx_data
    del approx_data["parameters"][param]
    
    # Delete the parameter from the scan_order in approx_data if it exists
    approx_data["scan_order"].pop(param, None)
    
    # Delete the parameter from the qp in approx_info if it exists
    approx_info.get("qp", {}).pop(param, None)
    
    # Delete the parameter from the dq_flag in approx_info if it exists
    approx_info.get("dq_flag", {}).pop(param, None)
    
def init_approx_data(parameters, model_info, qp_density, scan_order): 
    """
    Initialize approximation data based on input parameters.
    
    Args:
    - parameters (dict): A dictionary of parameters.
    - model_info (dict): A dictionary containing model information.
    - qp_density (int): The density of the quantization parameter.
    - scan_order (int): The order of scanning.
    
    Returns:
    - approx_data (dict): A dictionary containing initialized approximation data.
    """
    # Initialize approx_data dictionary with the given parameters
    approx_data = {
        "approx_method": {},
        "qp_density": np.int32(qp_density),
        "qp": {},
        "dq_flag": {},
        "decomposition_rank": {},
        "g_number_of_rows": {},
        "scan_order": {},
        "parameters": copy.copy(parameters),
        "compressed_parameter_types": {},
        "codebooks": {},
        "codebooks_egk": {},
        "codebook_zero_offsets": {},
    }
    
    # Loop over all parameters and update scan_order if necessary
    for x in parameters:
        
        # Check if parameter ends with _G or _H
        assert (x.endswith("_G") or x.endswith("_H")) == (("_G" in x) or ("_H" in x))
        if x.endswith("_G") or x.endswith("_H"):
            if len(model_info["parameter_dimensions"][x[:-2]]) > 1:
                approx_data["scan_order"][x] = np.int32(scan_order) 
        elif len(model_info["parameter_dimensions"][x]) > 1:
            approx_data["scan_order"][x] = np.int32(scan_order) 
        else:
            continue

    # Loop over all block identifiers in the model info
    for block_id in model_info["block_identifier"].values():
        
        # If block_id is not None, update the compressed_parameter_types
        if block_id != None:
            block_access = NNRBlockAccess(model_info, block_id)
            cpt = 0
            
            if block_access.bn_gamma:
                cpt += hls.BlockParameterTypes.NNR_CPT_BN
                
            if block_access.bi in approx_data["parameters"]: 
                cpt += hls.BlockParameterTypes.NNR_CPT_BI
                
            if block_access.dc_g in approx_data["parameters"]: 
                cpt += hls.BlockParameterTypes.NNR_CPT_DC
                par_dc_g = block_access.dc_g
                #par_dc_h = block_access.dc_h
                approx_data["decomposition_rank"][block_id] = approx_data["parameters"][par_dc_g].shape[1] 
                approx_data["g_number_of_rows"][block_id] = approx_data["parameters"][par_dc_g].shape[0] 
                
            if block_access.ls in approx_data["parameters"]: 
                cpt += hls.BlockParameterTypes.NNR_CPT_LS
                
            approx_data["compressed_parameter_types"][block_id] = cpt

    return approx_data

        
def fold_bn(model_info, approx_data, ap_info):
    """
    Fold Batch Normalization (BN) parameters in the model.

    Args:
    - model_info (dict): A dictionary containing model information.
    - approx_data (dict): A dictionary containing approximation data.
    - ap_info (object): An object containing approximation info.
    """
    # Access model using NNRModelAccess
    model_access = NNRModelAccess(model_info)
    
    # Loop over all blocks and parameters in the model
    for block_access in model_access.blocks_and_params():
        block_id = block_access.block_id
        
        if block_id is None:
            continue

        # Get the compressed parameter types and parameters for the current block
        cpt = approx_data["compressed_parameter_types"][block_id]
        ad = approx_data["parameters"]
        
        # Make sure that no approximation method is set
        assert not approx_data["approx_method"]
        
        # Define the epsilon value based on the model's topology storage format
        eps = 1e-3 if model_info['topology_storage_format'] == nnc_core.nnr_model.TopologyStorageFormat.NNR_TPL_TEF else 1e-5
        
        
        # Check if Batch Normalization (BN) compression is applied to the current block
        if cpt & hls.BlockParameterTypes.NNR_CPT_BN != 0:
            delta = block_access.bi
            bn_shape = ad[block_access.bn_mean].shape
            dq_flag = ap_info.approx_info["dq_flag"][block_access.bn_mean]
            
            # Check if delta parameter exists or needs to be created
            assert (cpt & hls.BlockParameterTypes.NNR_CPT_BI == 0) == (delta not in ad)
            
            if cpt & hls.BlockParameterTypes.NNR_CPT_BI == 0:
                
                # Create delta parameter and update compressed_parameter_types
                ad[delta] = np.zeros(bn_shape, dtype=np.float32)
                approx_data["compressed_parameter_types"][block_id] += hls.BlockParameterTypes.NNR_CPT_BI
                
                # Update quantization parameter (QP) and quantization flag for delta if using uniform approximation method
                if ap_info.approx_info["approx_method"] == "uniform":
                    ap_info.approx_info["qp"][delta] = ap_info.qp_other
                    ap_info.approx_info["dq_flag"][delta] = dq_flag
                    
            alpha = block_access.ls
            
            # Check if alpha parameter exists or needs to be created
            assert (cpt & hls.BlockParameterTypes.NNR_CPT_LS == 0) == (alpha not in ad)
            
            if cpt & hls.BlockParameterTypes.NNR_CPT_LS == 0:
                
                # Create alpha parameter and update compressed_parameter_types
                assert bn_shape == ad[block_access.bn_mean].shape
                
                ad[alpha] = np.ones(bn_shape, dtype=np.float32)
                approx_data["compressed_parameter_types"][block_id] += hls.BlockParameterTypes.NNR_CPT_LS
                
                # Update quantization parameter (QP) and quantization flag for alpha if using uniform approximation method
                if ap_info.approx_info["approx_method"] == "uniform":
                    ap_info.approx_info["qp"][alpha] = ap_info.qp_lsa
                    ap_info.approx_info["dq_flag"][alpha] = dq_flag

            # Calculate gamma parameter for folding
            g = ad[block_access.bn_gamma] / np.sqrt( ad[block_access.bn_var] + eps )
            
            # Delete gamma and variance parameters from approx_data and ap_info.approx_info
            del_param(approx_data, ap_info.approx_info, block_access.bn_gamma)
            del_param(approx_data, ap_info.approx_info, block_access.bn_var)
            
            # Fold BN parameters: update alpha and delta
            ad[alpha] *= g
            ad[delta] = (ad[delta] - ad[block_access.bn_mean]) * g + ad[block_access.bn_beta]
            
            # Delete mean and beta parameters from approx_data and ap_info.approx_info
            del_param(approx_data, ap_info.approx_info, block_access.bn_mean)
            del_param(approx_data, ap_info.approx_info, block_access.bn_beta)
            
            # Update compressed_parameter_types to indicate BN compression is applied
            approx_data["compressed_parameter_types"][block_id] -= hls.BlockParameterTypes.NNR_CPT_BN

            
def unfold_bn(model_info, approx_data):
    """
    Unfold Batch Normalization (BN) parameters in the model.

    Args:
    - model_info (dict): A dictionary containing model information.
    - approx_data (dict): A dictionary containing approximation data.
    """
    
    # Access model using NNRModelAccess
    model_access = NNRModelAccess(model_info)
    ad = approx_data["parameters"]
    
    # Loop over all blocks and parameters in the model
    for block_access in model_access.blocks_and_params():
        block_id = block_access.block_id
        
        if block_id is None:
            continue

        # Check if BN is absent in compressed_parameter_types but present in model_info
        bn_absent = approx_data["compressed_parameter_types"][block_id] & hls.BlockParameterTypes.NNR_CPT_BN == 0
        bn_folded = bn_absent and (block_access.bn_gamma in model_info["parameter_type"])
        
        if bn_folded:
            
            # Add BN back to compressed_parameter_types
            approx_data["compressed_parameter_types"][block_id] += hls.BlockParameterTypes.NNR_CPT_BN
            
            #ad = approx_data["parameters"]
            
            # Get delta parameter name and dimensions
            delta = block_access.bi
            dims = approx_data["parameters"][delta].shape
            
            if delta not in model_info["parameter_type"]:
                # Delta parameter was previously folded, move its values to bn_beta and delete delta
                assert approx_data["compressed_parameter_types"][block_id] & hls.BlockParameterTypes.NNR_CPT_BI != 0
                approx_data["parameters"][block_access.bn_beta] = approx_data["parameters"][delta]
                del approx_data["parameters"][delta]
                approx_data["compressed_parameter_types"][block_id] -= hls.BlockParameterTypes.NNR_CPT_BI
                
            else:
                # Create bn_beta parameter filled with zeros
                approx_data["parameters"][block_access.bn_beta] = np.zeros(dims, dtype=np.float32)

            # Create bn_mean, bn_gamma, and bn_var parameters filled with zeros and ones respectively
            approx_data["parameters"][block_access.bn_mean]  = np.zeros(dims, dtype=np.float32)
            approx_data["parameters"][block_access.bn_gamma] = np.ones(dims, dtype=np.float32)
            approx_data["parameters"][block_access.bn_var]   = np.ones(dims, dtype=np.float32)

def set_lsa(model_info, approx_data, lsa_params):
    """
    Set Layer-wise Scaling Approximation (LSA) parameters.

    Args:
    - model_info (dict): A dictionary containing model information.
    - approx_data (dict): A dictionary containing approximation data.
    - lsa_params (dict): A dictionary containing LSA parameters.
    """
    # Loop over all LSA parameters
    for k, v in lsa_params.items():

        # Set each parameter in approx_data
        approx_data["parameters"][k] = v.reshape([v.shape[0]])
        
        # Update compressed_parameter_types if necessary
        bi = model_info["block_identifier"].get(k, None)
        
        if bi is not None:
            approx_data["compressed_parameter_types"][bi] |= hls.BlockParameterTypes.NNR_CPT_LS

def apply_lsa(model_info, approx_data):
    """
    Apply Layer-wise Scaling Approximation (LSA) to the model.

    Args:
    - model_info (dict): A dictionary containing model information.
    - approx_data (dict): A dictionary containing approximation data.
    """
    # Assert approx_method is not set
    assert not approx_data["approx_method"]
    
    # Access model using NNRModelAccess
    model_access = NNRModelAccess(model_info)
    
    # Loop over all blocks and parameters in the model
    for block_access in model_access.blocks_and_params():
        block_id = block_access.block_id
        
        if block_id is None:
            continue
        
        # Check if the block has LSA parameters
        cpt = approx_data["compressed_parameter_types"][block_id]
        
        if cpt & hls.BlockParameterTypes.NNR_CPT_LS != 0:
            
            # Pop the LS parameter from approx_data and remove its entry from model_info
            ls = approx_data['parameters'].pop(block_access.ls)
            _ = model_info["parameter_index"].pop(block_access.ls, None)
            _ = model_info["block_identifier"].pop(block_access.ls, None)

            # Determine the weight parameter based on the block's compressed_parameter_types
            if cpt & hls.BlockParameterTypes.NNR_CPT_DC != 0:
                w = approx_data['parameters'][block_access.dc_g]
            else:
                w = approx_data['parameters'][block_access.w]
                
            # Reshape the LS parameter to match the dimensions of the weight parameter
            dims_ls = [-1] + [1] * (w.ndim - 1)
            w *= ls.reshape(dims_ls)
            
            # Remove the LSA flag from the compressed_parameter_types
            approx_data['compressed_parameter_types'][block_id] -= hls.BlockParameterTypes.NNR_CPT_LS

def recompose_params(model_info, approx_data_in):
    """
    Recompose parameters for the model.

    Args:
    - model_info (dict): A dictionary containing model information.
    - approx_data_in (dict): A dictionary containing input approximation data.

    Returns:
    - approx_data_out (dict): A dictionary containing output approximation data.
    """
    # Assert that no approximation method is set
    assert not approx_data_in["approx_method"]
    
    # Create a copy of approx_data_in
    approx_data_out = {k: copy.copy(v) for k, v in approx_data_in.items()} # create copies of dicts in approx_data

    # Access model using NNRModelAccess
    model_access = NNRModelAccess(model_info)
    
    # Loop over all blocks and parameters in the model
    for block_access in model_access.blocks_and_params():
        block_id = block_access.block_id
        
        if block_id is None:
            continue
        
        # Check if the block has decomposed parameters
        cpt = approx_data_out["compressed_parameter_types"][block_id]
        
        if cpt & hls.BlockParameterTypes.NNR_CPT_DC != 0:
            
            # Extract the decomposed parameters
            g = approx_data_out["parameters"].pop(block_access.dc_g)
            h = approx_data_out["parameters"].pop(block_access.dc_h)

            # Perform parameter recomposition
            recomposed_w = g.dot(h)
            recomposed_w = recomposed_w.reshape(model_info["parameter_dimensions"][block_access.w])

            # Update the recomposed parameters in approx_data_out
            approx_data_out["parameters"][block_access.w] = recomposed_w
            approx_data_out['compressed_parameter_types'][block_id] -= hls.BlockParameterTypes.NNR_CPT_DC

            # Update the parameter index in model_info
            param_id_g = model_info["parameter_index"][block_access.dc_g].pop()
            model_info["parameter_index"][block_access.w] = param_id_g
            
            # Remove unnecessary entries from model_info
            del(model_info["block_identifier"][block_access.dc_g])
            del(model_info["parameter_index"][block_access.dc_h])
            del(model_info["block_identifier"][block_access.dc_h])

    # Create a dictionary to store the recomposed parameters in the sorted order
    resorted_param_dict = dict()
    resorted_param_id_dict = {k: v for k, v in sorted(model_info["parameter_index"].items(), key=lambda item: item[1])}
    
    # Copy the recomposed parameters in the sorted order
    for param in resorted_param_id_dict.keys():
        resorted_param_dict[param] = copy.deepcopy(approx_data_out["parameters"][param])

    # Update the parameters in approx_data_out with the sorted recomposed parameters
    approx_data_out["parameters"] = resorted_param_dict

    return approx_data_out


def inference_based_qp_opt( 
        approx_info,
        model_info,
        model_executer,
        approx_data,
        param_opt,
        cabac_unary_length_minus1,
        verbose,
    ):
    
    """
    Perform inference-based Quantization Parameter (QP) optimization.

    Args:
    - approx_info (object): Contains approximation information.
    - model_info (dict): Contains model information.
    - model_executer (object): Executes the model.
    - approx_data (dict): Contains approximation data.
    - param_opt (bool): A flag for parameter optimization.
    - cabac_unary_length_minus1 (int): Length of Unary CABAC minus 1.
    - verbose (bool): If True, prints verbose output. Defaults to True.

    Returns:
    - Updates approx_info in-place.
    """
    
    # Perform approximation and record it
    approx_data_qp = approx(approx_info, model_info, approx_data, param_opt)
    rec_approx_data_qp = copy.deepcopy(approx_data_qp)
    rec(rec_approx_data_qp)
    
    # Start timer for encoding
    start = timer()
    __print_output_line("\tIOQ: PROCESSING QP FOR ALL TENSORS...", verbose=verbose) 
    enc_info_qp = {
        "cabac_unary_length_minus1" : cabac_unary_length_minus1,
        "param_opt_flag" : param_opt,
    }
    bitstream_qp = nnc_core.coder.encode(enc_info_qp, model_info, approx_data_qp)

    acc_qp = model_executer.eval_model(
        rec_approx_data_qp["parameters"],
        False,
    )
    
    refBSSize = len(bitstream_qp)
    refAcc = acc_qp[0]

    bestCost = 0.0
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose) 


    ############################ eval with QP-1
    start = timer()
    __print_output_line("\tIOQ: PROCESSING QP-1 FOR ALL TENSORS...", verbose=verbose) 
 
    approx_info_qp = copy.deepcopy(approx_info)

    for p in approx_info_qp["qp"].keys():
        if model_info["parameter_type"][p] in nnc_core.nnr_model.W_TYPES:
            approx_info_qp["qp"][p] -= 1

    approx_data_qp = approx(
        approx_info_qp,
        model_info,
        approx_data,
        param_opt,
    )
    rec_approx_data_qp = copy.deepcopy(approx_data_qp)
    rec(
        rec_approx_data_qp,
    )
    
    ##encode
    enc_info_qp = {
        "cabac_unary_length_minus1" : cabac_unary_length_minus1,
        "param_opt_flag" : param_opt,
    }
    bitstream_qp = nnc_core.coder.encode(enc_info_qp, model_info, approx_data_qp)
    ##eval
    acc_qp = model_executer.eval_model(
        rec_approx_data_qp["parameters"],
        False,
    )

    currBSSize = len(bitstream_qp)
    currAcc = acc_qp[0]

    diffBR = currBSSize - refBSSize
    diffAcc = refAcc - currAcc

    lambdaM1 = -diffAcc/diffBR 

    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)  


    ############################### eval with QP+1
    start = timer()
    __print_output_line("\tIOQ: PROCESSING QP+1 FOR ALL TENSORS...", verbose=verbose) 

    approx_info_qp = copy.deepcopy(approx_info)

    for p in approx_info_qp["qp"].keys():
        if model_info["parameter_type"][p] in nnc_core.nnr_model.W_TYPES:
            approx_info_qp["qp"][p] += 1

    approx_data_qp = approx(
        approx_info_qp,
        model_info,
        approx_data,
        param_opt,
    )
    rec_approx_data_qp = copy.deepcopy(approx_data_qp)
    rec(
        rec_approx_data_qp,
    )
    
    ##encode
    enc_info_qp = {
        "cabac_unary_length_minus1" : cabac_unary_length_minus1,
        "param_opt_flag" : param_opt,
    }
    bitstream_qp = nnc_core.coder.encode(enc_info_qp, model_info, approx_data_qp)
    ##eval
    acc_qp = model_executer.eval_model(
        rec_approx_data_qp["parameters"],
        False,
    )

    currBSSize = len(bitstream_qp)
    currAcc = acc_qp[0]

    diffBR = currBSSize - refBSSize
    diffAcc = refAcc - currAcc

    lambdaP1 = -diffAcc/diffBR
    
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose) 

    ################################

    ##sort parameters by size
    mapParamToSize = []
    approx_info_qp = copy.deepcopy(approx_info)
    for p in rec_approx_data_qp["parameters"]:
        if model_info["parameter_type"][p] in nnc_core.nnr_model.W_TYPES:
            mapParamToSize.append([p , np.size(approx_data_qp["parameters"][p])])
    
    mapParamToSize.sort(key = lambda x: x[1],reverse=True) 
    
    setNeg = [-4, -3, -2, -1 ]
    setPos = [1, 2, 3, 4 ]

    qpOffsetSet = [setNeg, setPos]

    timeLastQp = "n/a"

    for iParam, item in enumerate(mapParamToSize[1::]):
        for iQpSet, qpSet in enumerate(qpOffsetSet):
            for iQpOff, qp_off in enumerate(qpSet):
                __print_output_line("\r\tIOQ: PROCESSING TENSOR {}/{} AND QP {}/{} (LAST QP-ITERATION TOOK: {})".format( iParam+1, len(mapParamToSize)-1, iQpSet*len(qpOffsetSet[0])+ iQpOff + 1, len(qpOffsetSet[0])+len(qpOffsetSet[1]), timeLastQp), verbose=verbose)
                start = timer()
                approx_info_qp_curr = copy.deepcopy(approx_info_qp)
                approx_info_qp_curr["qp"][item[0]] = approx_info["qp"][item[0]] + qp_off

                approx_data_qp = approx(
                    approx_info_qp_curr,
                    model_info,
                    approx_data,
                    param_opt,
                )
                    
                rec_approx_data_qp = copy.deepcopy(approx_data_qp)
                rec(
                    rec_approx_data_qp,
                )
                
                ##encode
                enc_info_qp = {
                    "cabac_unary_length_minus1" : cabac_unary_length_minus1,
                    "param_opt_flag" : param_opt,
                }
                
                bitstream_qp = nnc_core.coder.encode(enc_info_qp, model_info, approx_data_qp)
            
                ##eval
                acc_qp = model_executer.eval_model(
                    rec_approx_data_qp["parameters"],
                    False,
                )
                
                currBSSize = len(bitstream_qp)
                currAcc = acc_qp[0]

                diffBR = currBSSize - refBSSize
                diffAcc = refAcc - currAcc

                lamb = max( (lambdaP1 + lambdaM1) / 2, 0.0 )

                currCost = diffAcc + lamb * diffBR

                if currCost < bestCost:
                    approx_info_qp = copy.deepcopy(approx_info_qp_curr)
                    bestCost = currCost
                
                end = timer()
                timeLastQp = "{:.4f} s".format( end-start )

    __print_output_line("\n")
    approx_info.clear()
    approx_info.update(approx_info_qp)

############################ Core Part of LSA #############################
def run_ft_and_lsa(model_info, approx_data, ap_info, model_executer, block_id_and_param_type, lsa_flag, ft_flag, use_dq, verbose, bitstream_path):

    # model_info : nnc_mdl.model_info,
    
    """
    approx_data : {
        "approx_method": {},
        "qp_density": np.int32(qp_density),
        "qp": {},
        "dq_flag": {},
        "decomposition_rank": {},
        "g_number_of_rows": {},
        "scan_order": {},
        "parameters": copy.copy(parameters),
        "compressed_parameter_types": {},
        "codebooks": {},
        "codebooks_egk": {},
        "codebook_zero_offsets": {},
        }
    """
    
    # approximation information  
    """                                          
    ap_info : {
        "approx_method", # codebook or other methods
        "codebook_mode",
        "dq_flag",
        "lambda_scale",
        "cabac_unary_length_minus1",
        "to_approximate", # candidate parameter types to approximate
        }
    """
    
    """
    block_id_and_param_type:
    
        {key name of state_dict : custom name used in this NNCodec}
        
        ex) {'model.rgb_linear.weight' : 'weight' ...}
    """
    
    # Copy original approx_info
    # And do quantization - dequantization - find lsa parameter by tune_model
    # Apply lsa parameter to original approx_data
    approx_info_ft = copy.deepcopy(ap_info.approx_info)
    
    if not lsa_flag:
        # If LSA flag is False, include only weight parameters for approximation
        approx_info_ft["to_approximate"] = W_TYPES # W_TYPES = ["weight"]
    else:
        # If LSA flag is True, 
        # remove 'weight.ls' (= lsa param) from the parameters to approximation
        approx_info_ft["to_approximate"].remove('weight.ls')
    
    # Quantization
    approx_data_ft = nnc_core.approximator.approx(approx_info_ft, model_info, approx_data)
    
    # De-Quantization
    nnc_core.approximator.rec(approx_data_ft)

    # Tune the model using the forward transformed parameters
    tuned_params = model_executer.tune_model(
        bitstream_path, # Only for save
        parameters=approx_data_ft['parameters'],
        param_types=model_info['parameter_type'],
        lsa_flag=lsa_flag,
        ft_flag=ft_flag,
        verbose=verbose,
    )
    
    lsa_params = tuned_params[0]
    ft_params  = tuned_params[1]

    if ft_flag:
        # Update the original parameters with fine-tuned parameters if FT flag is True
        approx_data["parameters"].update(ft_params)
        
    if lsa_flag:
        if block_id_and_param_type:
            nnc_core.approximator.set_lsa(model_info, approx_data, lsa_params)
            nnc_core.nnr_model.add_lsa_to_block_id_and_param_type( block_id_and_param_type, lsa_params )
        else:
            approx_data["parameters"].update(lsa_params)
            
        ap_info.set_ls_qps(model_info, approx_data, 1 if use_dq else 0)


def approx(approx_info, model_info, approx_data, param_opt=0):
    approx_method = approx_info['approx_method']
    
    # Perform skip approximation for integer parameters
    approx_data = integer.skip_approx( approx_info, model_info, approx_data )

    if approx_method == 'codebook':
        # Perform codebook approximation for codebook method
        approx_data, approx_info = codebook.approx(approx_info, model_info, approx_data, param_opt)

    # Perform baseline approximation for other methods
    return baseline.approx(approx_info, model_info, approx_data)
    

def rec(approx_data):
    for param in approx_data['parameters']:
        if param in approx_data["approx_method"]:
            
            if approx_data["approx_method"][param] == 'uniform':
                # Perform reconstruction for uniform approximation
                baseline.rec(param, approx_data)
                
            elif approx_data["approx_method"][param] == 'codebook':
                # Perform reconstruction for codebook approximation
                codebook.rec(param, approx_data)
                
            elif approx_data["approx_method"][param] == 'skip':
                # Perform reconstruction for skip approximation
                integer.skip_rec(param, approx_data)
                
            else:
                assert param not in approx_data["approx_method"], "unknown approx_method"


class ApproxInfo():
    def __init__(
        self,
        approx_data,
        model_info,
        approx_method,
        codebook_mode,
        qp,
        opt_qp,
        disable_dq,
        cabac_unary_length_minus1,
        lambda_scale,
        nonweight_qp=None,
        qp_per_tensor=None,
    ):
        """
        Initializes an ApproxInfo object.

        Args:
            approx_data (dict): Approximation data.
            model_info (dict): Model information.
            approx_method (str): Approximation method.
            codebook_mode (int): Codebook mode.
            qp (int): QP value.
            opt_qp (bool): Flag indicating whether to optimize QP values.
            disable_dq (bool): Flag indicating whether to disable DQ.
            cabac_unary_length_minus1 (int): CABAC unary length minus 1.
            lambda_scale (float): Lambda scaling factor.
            nonweight_qp (int, optional): QP value for non-weight parameters. Defaults to None.
            qp_per_tensor (dict, optional): QP values per tensor. Defaults to None.
        """
    
        # Initialize the approximate information dictionary
        self.__approx_info = {
            "approx_method": "codebook" if codebook_mode > 0 else approx_method,
            "codebook_mode": codebook_mode,
            "dq_flag": {x: 0 if disable_dq else 1 for x in approx_data["parameters"]},
            "lambda_scale": lambda_scale,
            "cabac_unary_length_minus1": cabac_unary_length_minus1,
            "to_approximate": nnc_core.nnr_model.W_TYPES + nnc_core.nnr_model.O_TYPES,
        }

        if approx_method == "uniform" or approx_method == "codebook":
            qp = np.int32(qp)
            qp_density = approx_data["qp_density"]
            
            # Calculate QP values for different types of parameters
            self.__qp_other = nonweight_qp if nonweight_qp else qp - (2 << qp_density)  # same as dividing the stepsize by 4
            self.__qp_lsa = nonweight_qp if nonweight_qp else qp - (2 << qp_density)#qp - (8 << qp_density)
            self.approx_info["qp"] = {} # Initialize the QP dictionary
            
            for x in approx_data["parameters"]:
                if x not in model_info["parameter_index"] and (x.endswith("_G") or x.endswith("_H")):
                    assert model_info["parameter_type"][x[:-2]] in nnc_core.nnr_model.W_TYPES, "Unexpected."
                    self.approx_info["qp"][x] = qp # Set QP value for weight parameters
                else:
                    self.approx_info["qp"][x] = qp if model_info["parameter_type"][x] in nnc_core.nnr_model.W_TYPES else self.qp_other
                    
            if qp_per_tensor is not None:
                assert type(qp_per_tensor) is dict, "qp_per_tensor must be a dict!"  
                for x in approx_data["parameters"]:
                    self.approx_info["qp"][x] = qp_per_tensor.get(x, self.approx_info["qp"][x])
                    
            if opt_qp:
                self._modify_qp(approx_data, model_info) # Optimize QP values based on parameter statistics

    @property
    def qp_lsa(self):
        return self.__qp_lsa # Get the QP value for LSA
 
    @property
    def qp_other(self):
        return self.__qp_other # Get the QP value for non-weight parameters

    @property
    def approx_info(self):
        return self.__approx_info # Get the approximate information dictionary

    def apply_qp(self, approx_data, model_info, qp, nonweight_qp=None):
        """
        Apply QP values to the approximate data.

        Args:
            approx_data (dict): Approximation data.
            model_info (dict): Model information.
            qp (int): QP value.
            nonweight_qp (int, optional): QP value for non-weight parameters. Defaults to None.
        """
        qp = np.int32(qp)
        qp_density = approx_data["qp_density"]
        
        self.__qp_other = nonweight_qp if nonweight_qp else qp - (2 << qp_density) # Set QP value for non-weight parameters
        self.__qp_lsa = nonweight_qp if nonweight_qp else qp - (2 << qp_density)#qp - (8 << qp_density)
        self.approx_info["qp"] = {}
        
        for x in approx_data["parameters"]:
            if x not in model_info["parameter_index"] and (x.endswith("_G") or x.endswith("_H")):
                assert model_info["parameter_type"][x[:-2]] in nnc_core.nnr_model.W_TYPES, "Unexpected."
                # Set QP value for weight parameters
                self.approx_info["qp"][x] = qp 
            else:
                if model_info["parameter_type"][x] in nnc_core.nnr_model.W_TYPES:
                    # Set QP value for weight parameters
                    self.approx_info["qp"][x] = qp 
                else:
                    # Set QP value for other parameter types
                    self.approx_info["qp"][x] = self.qp_other 
    
    def _modify_qp(self, approx_data, model_info):
        """
        Modify QP values based on parameter statistics.

        Args:
            approx_data (dict): Approximation data.
            model_info (dict): Model information.
        """
        param_types = ["weight"]  # Parameter types to consider for QP modification
        param_names = []  # Names of the parameters to modify QP
        param_sizes = []  # Sizes of the parameters
        param_std = []  # Standard deviations of the parameters
        
        for k, v in approx_data["parameters"].items():
            param_w = k[:-2] if ( k.endswith("_G") or k.endswith("_H") ) else k 
            
            if model_info["parameter_type"][param_w] in param_types:
                if ( k.endswith("_G") or k.endswith("_H") ):
                    if k.endswith("_G") : continue
                    assert k.endswith("_H")
                    
                    g = approx_data["parameters"][param_w + "_G"].shape
                    h = approx_data["parameters"][param_w + "_H"].shape
                    
                    if len( h ) == 4:
                        assert h[0] == 1
                        assert h[1] == 1
                        
                    assert h[-2] == g[-1]
                    s = np.prod(g[:-1]) * h[-1]
                    
                    param_names.append(param_w + "_G")
                    param_sizes.append(0)
                    param_std.append(0)
                    param_names.append(param_w + "_H")
                    param_sizes.append(s)
                    param_std.append(np.std(np.concatenate(
                        (approx_data["parameters"][param_w + "_G"].flatten(),
                         approx_data["parameters"][param_w + "_H"].flatten()), axis=0)))
                else:
                    param_names.append(k)
                    param_sizes.append(v.size)
                    param_std.append(np.std(v))

        rel_layer_sizes = np.array(param_sizes) / sum(param_sizes)  # Calculate relative sizes of the layers
        rel_layer_std = np.array(param_std) / max(param_std)  # Calculate relative standard deviations of the layers

        shares = rel_layer_sizes + (.1 * (1 - rel_layer_std))

        w = dict(zip(param_names, shares))  # Create a dictionary of layer shares
        
        for name in param_names:
            qp = self.__approx_info['qp'][name]  # Get the initial QP value for the parameter

            # Cap the share value at 0.15 to prevent excessive reduction
            if w[name] > .5: 
                w[name] = .15
                
            self.__approx_info['qp'][name] = np.int32(round(qp * (1 - w[name])))
            
            if name.endswith( "_H" ):
                self.__approx_info['qp'][name[:-2]+"_G"] = self.__approx_info['qp'][name]

    def set_ls_qps(self, model_info, approx_data, dq_flag):
        """
        Set QP and DQ flags for LS parameters.

        Args:
            model_info (dict): Model information.
            approx_data (dict): Approximation data.
            dq_flag (int): DQ flag value.
        """
        for block_access in NNRModelAccess(model_info).blocks_and_params():
            if block_access.block_id is not None:
                
                cpt = approx_data["compressed_parameter_types"][block_access.block_id]
                
                if cpt & hls.BlockParameterTypes.NNR_CPT_LS != 0:
                    # Set QP and DQ flags for LS parameters
                    self.approx_info["qp"][block_access.ls] = self.qp_lsa
                    self.approx_info["dq_flag"][block_access.ls] = dq_flag
            

