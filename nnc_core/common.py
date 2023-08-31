import numpy as np
    
def get_qp_from_stepsize( stepsize, qp_density ):
    """
    Calculates the Quantization Parameter (QP) given a stepsize and a density.
    
    The calculation is based on a logarithmic scale, which is common for quantization processes in video compression.
    
    Args:
    - stepsize (float): The stepsize to be used in the quantization process.
    - qp_density (int): The density of the quantization parameter.
    
    Returns:
    - qp (float): The calculated Quantization Parameter.
    """
    
    k = 1 << qp_density # = 2 ** qp_density

    # Base QP is calculated by taking the floor of log base 2 of stepsize, multiplied by k
    baseQP = np.floor( np.log2( stepsize ) ) * k
    
    # QP is calculated using the formula given
    qp = baseQP + ( (stepsize * k) / 2**(baseQP/k) - k )

    # Returns the calculated QP
    return qp

def get_stepsize_from_qp( qp, qp_density ):
    """
    Calculates the stepsize given a Quantization Parameter (QP) and a density.
    
    This function appears to reverse the operation performed in the 'get_qp_from_stepsize' function.
    
    Args:
    - qp (int/float): The Quantization Parameter to be used.
    - qp_density (int): The density of the quantization parameter.
    
    Returns:
    - delta (float): The calculated stepsize.
    """
    
    k = 1 << qp_density # = 2 ** qp_density
    mul = k + (qp & (k-1)) # & : bitwise and operator
    shift = qp >> qp_density # = floor(qp / 2 ** qp_density)
    delta = mul * (2.0 ** (shift - qp_density))
    return delta

def compute_qp_offset_to_dq_equivalent( qp_density ):
    """
    Calculates the QP offset for a given QP density.
    
    This offset seems to be used in some sort of adjustment or correction during the quantization process.
    
    Args:
    - qp_density (int): The density of the quantization parameter.
    
    Returns:
    - qp_off (int): The calculated QP offset.
    """
    qp_off = (1 << qp_density) # Shift operator to calculate the value of 2 raised to the power of qp_density

    return qp_off # Returns the calculated QP offset
