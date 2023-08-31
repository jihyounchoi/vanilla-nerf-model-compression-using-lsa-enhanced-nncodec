from nnc_core import hls
import numpy as np


def encode(encoder, approx_data, param, ndu, mps, lps, param_opt_flag):
    if (
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_FLOAT) or
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK)
    ):
        quantization_parameter =  lps["lps_quantization_parameter"] if lps is not None else mps["mps_quantization_parameter"]
        qp_density             =  lps["lps_qp_density"] if lps is not None else mps["mps_qp_density"]
        encoder.iae_v( 6 + qp_density, approx_data["qp"][param] - quantization_parameter) 
    
    encoder.initCtxModels( ndu["cabac_unary_length_minus1"], param_opt_flag )
    if param in approx_data["scan_order"]:
        assert ndu["scan_order"] == approx_data["scan_order"][param], "All parameters of a block must use the same scan_order."
    scan_order = ndu.get("scan_order", 0)
    if approx_data["parameters"][param].ndim <= 1:
        scan_order = 0
    encoder.encodeLayer(approx_data["parameters"][param], approx_data["dq_flag"][param], scan_order)


def decode( decoder, approx_data, param, ndu, mps, lps ):
    if (
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_FLOAT) or
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK)
    ):
        quantization_parameter        =  lps["lps_quantization_parameter"] if lps is not None else mps["mps_quantization_parameter"]
        qp_density                    =  lps["lps_qp_density"] if lps is not None else mps["mps_qp_density"]
        approx_data["qp"][param]      = np.int32(decoder.iae_v( 6 + qp_density ) + quantization_parameter)
        approx_data["dq_flag"][param] = ndu["dq_flag"]
    
    else:
        approx_data["dq_flag"][param] = 0
        
    decoder.initCtxModels( ndu["cabac_unary_length_minus1"] )
    scan_order = ndu.get("scan_order", 0)
    if approx_data["parameters"][param].ndim <= 1:
        scan_order = 0
    decoder.decodeLayer(approx_data["parameters"][param], approx_data["dq_flag"][param], scan_order)


def decodeAndCreateEPs( decoder, approx_data, param, ndu, mps, lps ):
    if (
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_FLOAT) or
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK)
    ):
        quantization_parameter        = lps["lps_quantization_parameter"] if lps is not None else mps["mps_quantization_parameter"]
        qp_density                    = lps["lps_qp_density"] if lps is not None else mps["mps_qp_density"]
        approx_data["qp"][param]      = np.int32(decoder.iae_v( 6 + qp_density ) + quantization_parameter)
        approx_data["dq_flag"][param] = ndu["dq_flag"]
        
    decoder.initCtxModels( ndu["cabac_unary_length_minus1"] )
    scan_order = ndu.get("scan_order", 0)
    if approx_data["parameters"][param].ndim <= 1:
        scan_order = 0
    entryPointArray = decoder.decodeLayerAndCreateEPs(approx_data["parameters"][param], approx_data.get("dq_flag", {}).get(param, 0), scan_order)

    return entryPointArray


