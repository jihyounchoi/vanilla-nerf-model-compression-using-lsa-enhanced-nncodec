import copy
import enum
from abc import ABC, abstractmethod
import nnc_core
import numpy as np
from nnc_core import hls, common
import pickle

# Define enumerations for different types of model formats and compression formats
class TopologyStorageFormat(enum.IntEnum):
    NNR_TPL_UNREC   = 0
    NNR_TPL_NNEF    = 1
    NNR_TPL_ONNX    = 2
    NNR_TPL_PYT     = 3
    NNR_TPL_TEF     = 4
    NNR_TPL_PRUN    = 5
    NNR_TPL_REFLIST = 6
    
class TopologyCompressionFormat(enum.IntEnum):
    NNR_PT_RAW = 0
    NNR_DFL    = 1

# Define the parameter types for weights and other parameters
# Weight types
W_TYPES = [
    "weight" 
]

# Not Weight Types
O_TYPES = [
    "weight.ls",
    "bias",
    "bn.beta",
    "bn.gamma",
    "bn.mean",
    "bn.var",
    "unspecified"
]

# Define an Abstract Base Class (ABC) for a model execution class (Due to not implemented issue)
# This class provides the structure for model execution tasks but does not implement them
class ModelExecute(ABC):
    
    # eval_model, test_model, tune_model not yet implemented lololololol
    def eval_model(self,
                   parameters,
                   verbose=False,
                   ):
        
        raise NotImplementedError(
            "Function eval_model not yet implemented."
            "This function is e.g. required for Inference-optimised quantization (IOQ)."
            "Either implement eval_model or deactivate IOQ (controlled by parameter ioq)!"
            )


    def test_model(self,
                   parameters,
                   verbose=False,
                   ):
        
        raise NotImplementedError(
            "Function test_model not yet implemented."
            "This function is e.g. required for inference."
            )
    
    
    def tune_model(self,
                   parameters,
                   param_types,
                   lsa_flag,
                   ft_flag,
                   verbose=False,
                   ):
        
        raise NotImplementedError(
            "Function tune_model not yet implemented. "
            "This function is required for fine tuning (FT) and local scaling adaptation (LSA)."
            "Either implement the function or deactivate fine tuning (controlled by parameter ft)"
            "and local scaling adaptation (controlled by parameter lsa)!"
            )
    
    # Just returns False on current version
    @abstractmethod
    def has_eval(self):
        return False
    
    @abstractmethod
    def has_test(self):
        return False
    
    @abstractmethod
    def has_tune_ft(self):
        return False
    
    @abstractmethod
    def has_tune_lsa(self):
        return False
    
    
def create_NNC_model_instance_from_file( model_path ):
    """
    This function reads a model file from disk and creates an instance of the NNRModel.
    
    Args:
        model_path (str): The path to the model file.
        
    Returns:
        NNRModel: An instance of NNRModel.
        parameter_dict: A dictionary that contains model parameters.
        
    Raises:
        SystemExit: If the model file cannot be read or if the model parameters are not in the expected format.
    """
    try:
        # Attempt to read the model file
        parameter_dict = pickle.load( model_path )
        
        """ Check if the parameter dictionary is in the correct format """
        if (
            isinstance(parameter_dict, dict) and # checking parameter_dict is dictionary type
            all( [isinstance(a, np.ndarray) for a in parameter_dict.values()] ) and # all values in parameter_dict are np.ndarray
            (
                # all values in parameter_dict have a type of float32 or int32
                all([a.dtype==np.float32 for a in parameter_dict.values()]) or
                all([a.dtype==np.int32 for a in parameter_dict.values()])
                )
            ):
            
            # Create an instance of NNRModel
            NNCModel = NNRModel( parameter_dict )
        
        else:
            raise SystemExit("Parameter dict must be a dict (key-value pairs). The keys shall be stings, specifying the tensor names. The values must be numpy arrays (ndarray) of type float32 or int32!")
    except:
        raise SystemExit("Can't read model: {}".format(model_path))

    return NNCModel, parameter_dict


def save_to_pickled_file( model_dict, model_path ):
    """
    This function saves a model dictionary to a file using pickle.
    
    Args:
        model_dict (dict): The model dictionary to save.
        model_path (str): The path to the file where the model dictionary should be saved.
    """
    pickle.dump( model_dict, open(model_path, "wb") )


# The NNRModel class is used to represent a neural network model
# It includes methods to initialize the model from a dictionary of parameters,
# guess block ID and parameter types, and get model information

class NNRModel():
    
    """
    This class is used to represent a Neural Network model. 
    This includes methods for initializing the model from a dictionary of parameters 
    and a method stub for guessing block IDs and parameter types 
    (which throws an error in this generic class but can be implemented in specific model subclasses). 
    The class also has a model_info property that returns model information.
    """

    # The constructor of the NNRModel class
    # It can take an optional model_dict argument which is a dictionary of model parameters
    def __init__(self, model_dict=None):
        """
        Constructor for the NNRModel class.

        Args:
            model_dict (dict, optional): A dictionary of model parameters. Defaults to None.
        """
        
        # If a model_dict is provided and it is a dictionary, call the init_model_from_dict method
        if model_dict and isinstance(model_dict, dict):
            self.init_model_from_dict(model_dict)

        # Otherwise, set the model_info attribute to None
        else:
            self.__model_info = None

        self.model = None

    # The init_model_from_dict method initializes the model from a dictionary of parameters
    def init_model_from_dict(self, model_dict):
        """
        This method is used to initialize the model from a dictionary of parameters. 
        It processes the parameters and stores them along with related information 
        such as their type, dimensions, and index. 
        It also sets the model's topology storage and compression formats.

        Args:
            model_dict (dict): A dictionary of model parameters.

        Returns:
            dict: The parameters dictionary.

        Raises:
            SystemExit: If the model_dict is not of type dict.
        """

        # Check if model_dict is a dictionary
        if isinstance(model_dict, dict):
            model_dict = model_dict
            
        # If not, raise an exception
        else:
            raise SystemExit("model_dict must be of type dict")
        
        # Define a model_data dictionary that will store the parameters of the model
        # and the reduction method to be used for the model
        model_data = {'parameters': {}, 'reduction_method': 'baseline'}
        
        # Define a model_info dictionary that will store various information about the model
        model_info = {'parameter_type': {}, 'parameter_dimensions': {}, 'parameter_index': {}, 'block_identifier': {}, 'original_size': {}, 'topology_storage_format' : None, 'topology_compression_format' : None}

        # Define lists of different data types
        type_list_int = ['int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32']
        type_list_1_bytes = ['int8', 'uint8']
        type_list_2_bytes = ['int16', 'uint16', 'float16']

        original_size = 0

        # Iterate over the modules in the model_dict
        for i, module_name in enumerate(model_dict):

            # If the data type of the module is in type_list_1_bytes, add the size of the module to original_size
            if model_dict[module_name].dtype in type_list_1_bytes:
                original_size += model_dict[module_name].size

            # If the data type of the module is in type_list_2_bytes, add twice the size of the module to original_size
            elif model_dict[module_name].dtype in type_list_2_bytes:
                original_size += model_dict[module_name].size*2
                
            # Otherwise, add four times the size of the module to original_size
            else:
                original_size += model_dict[module_name].size*4
                
            # Convert the module to int32 if its data type is in type_list_int and store it in the parameters dictionary in model_data
            # Otherwise, just store the module in the parameters dictionary in model_data
            model_data['parameters'][module_name] = np.int32(model_dict[module_name]) if model_dict[module_name].dtype in type_list_int else model_dict[module_name]

            # Store the shape of the module in the parameter_dimensions dictionary in model_info
            mdl_shape = model_data['parameters'][module_name].shape
            model_info['parameter_dimensions'][module_name] = mdl_shape
            
            # If the shape of the module is 0 (i.e., the module is a scalar),
            # convert the module to float32 and store it in a one-element array,
            # and store the shape of the array in the parameter_dimensions dictionary in model_info
            if len(mdl_shape) == 0: # scalar
                model_data['parameters'][module_name] = np.array([np.float32(model_data['parameters'][module_name])])
                model_info['parameter_dimensions'][module_name] = np.array([0]).shape
                
            # Store the index of the module in the parameter_index dictionary in model_info
            model_info['parameter_index'][module_name] = i

            # Get the number of dimensions of the module
            dims = len(mdl_shape)

            # If the number of dimensions is greater than 1, store 'weight' in the parameter_type dictionary in model_info
            # Otherwise, store 'unspecified' in the parameter_type dictionary in model_info
            if dims > 1:
                model_info['parameter_type'][module_name] = 'weight'
            else:
                model_info['parameter_type'][module_name] = 'unspecified'

        # Store the topology storage format and the topology compression format in model_info
        model_info['topology_storage_format'] = nnc_core.nnr_model.TopologyStorageFormat.NNR_TPL_UNREC
        model_info['topology_compression_format'] = nnc_core.nnr_model.TopologyCompressionFormat.NNR_PT_RAW

        # Store the original size of the model in model_info
        model_info["original_size"] = original_size

        # Store model_info in the model_info attribute of the NNRModel instance
        self.__model_info = model_info

        # Return the parameters dictionary in model_data
        return model_data["parameters"]
    
    # The guess_block_id_and_param_type method raises an exception because block ID and parameter type cannot be guessed for the generic model class
    # This method should be overridden in a subclass that represents a specific model class (e.g., a PyTorch model or a TensorFlow model)
    def guess_block_id_and_param_type(self, model_parameters):
        """
        This method is a placeholder for a function 
        that would guess the block IDs and parameter types for the model. 
        In this generic class, it simply raises an error stating 
        that the method should be implemented in a subclass for a specific type of model.
        
        Args:
            model_parameters: The model parameters.

        Raises:
            SystemExit: Indicates that block ID and parameter type cannot be guessed for the generic model class.
                It advises providing a PyTorch model, a TensorFlow model, or block_id_and_param_type.
        """
        raise SystemExit("Block id and parameter type can not be guessed for generic model class. Try to provide a pytorch model, a tensorflow model or block_id_and_param_type (see description of compress_model)!")

    # The model_info property returns the model_info attribute of the NNRModel instance   
    @property
    def model_info(self):
        """
        Property that returns the model_info attribute of the NNRModel instance.

        Returns:
            dict: The model information.
        """
        return self.__model_info


class NNRParamAccess():
    """
    This class provides an interface for accessing parameters of a neural network model. 
    It includes a param_generator method for yielding parameters and a block_id property 
    that returns None (since individual parameters don't have a block ID), 
    and a param property that returns the parameter itself.
    """
    def __init__(self, model_info, param):
        """
        Initializes an instance of NNRParamAccess.

        Args:
            model_info (dict): Model information dictionary containing parameter types, dimensions, and block identifiers.
            param (str): Name of the parameter.
        """
        self.__single_param = (model_info["parameter_type"].get(param), param, model_info["parameter_dimensions"].get(param))

    def param_generator(self, cpt_dict_dummy):
        """
        Generator method that yields the single parameter.

        Args:
            cpt_dict_dummy: Unused argument.

        Yields:
            tuple: Contains the parameter type, parameter name, and parameter dimensions.
        """
        yield self.__single_param

    @property
    def block_id(self):
        """
        Property that returns the block identifier (None for individual parameters).

        Returns:
            None: Individual parameters don't have a block ID.
        """
        return None

    @property
    def param(self):
        """
        Property that returns the name of the parameter.

        Returns:
            str: Name of the parameter.
        """
        return self.__single_param[1]
    

class NNRBlockAccess():
    """
    This class provides an interface for accessing blocks of a neural network model.
    It includes properties for different types of block parameters 
    (e.g., weights, bias, batch normalization parameters), 
    and two generator methods for yielding parameters and topology elements based on the compression parameters.
    """
    def __init__(self, model_info, block_identifier):
        """
        Initialize the NNRBlockAccess object with the given model_info and block_identifier.

        Args:
        - model_info (dict): Model information dictionary containing parameter types, dimensions, and block identifiers.
        - block_identifier (str): Identifier of the block.

        """
        self.__block_identifier = block_identifier
        self.__model_info = model_info
    
        # Create a dictionary mapping parameter types to their names within the block
        block_list = [
            x
            for x in model_info["block_identifier"]
            if model_info["block_identifier"][x] == block_identifier
        ] 
        
        self.__block_dict = { model_info["parameter_type"][x]: x for x in block_list }


    @property
    def block_id(self):
        return self.__block_identifier

    @property
    def w(self):
        for x in ["weight"]:
            if x in self.__block_dict:
                return self.__block_dict[x]

    @property
    def dc_g(self):
        return self.w + "_G"

    @property
    def dc_h(self):
        return self.w + "_H"

    @property
    def ls(self):
        return self.w + "_scaling"
            
    @property
    def bn_beta(self):
        return self.__block_dict.get("bn.beta", None)

    @property
    def bn_gamma(self):
        return self.__block_dict.get("bn.gamma", None)

    @property
    def bn_mean(self):
        return self.__block_dict.get("bn.mean", None)

    @property
    def bn_var(self):
        return self.__block_dict.get("bn.var", None)

    @property
    def bi(self):
        for x in ["bias"]:
            if x in self.__block_dict:
                return self.__block_dict[x]
            
        for x in ["weight"]:
            if x in self.__block_dict:
                return self.__block_dict[x] + ".bias"

    
    def param_generator(self, compressed_parameter_types_dict):
        """
        Generator method that yields parameters based on the compression parameters.

        Args:
        - compressed_parameter_types_dict (dict): Dictionary containing compressed parameter types.

        Yields:
        - Tuple: Contains the parameter type, parameter name, and parameter dimensions.
        """
        compressed_parameter_types = compressed_parameter_types_dict[self.block_id]
        
        # These are property methods that return certain attributes of the block. 
        # The names of the properties are abbreviated, 
        # but they correspond to different types of parameters in the block, 
        # such as weights (w), bias (bi), batch normalization parameters (bn_beta, bn_gamma, bn_mean, bn_var), and others.
        
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_LS != 0: 
            yield "weight.ls", self.ls, [self.__model_info["parameter_dimensions"][self.w][0]]
            
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_BI != 0:
            yield "bias", self.bi, [self.__model_info["parameter_dimensions"][self.w][0]]
            
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_BN != 0:
            yield "bn.beta", self.bn_beta, self.__model_info["parameter_dimensions"][self.bn_beta]
            yield "bn.gamma", self.bn_gamma, self.__model_info["parameter_dimensions"][self.bn_gamma]
            yield "bn.mean", self.bn_mean, self.__model_info["parameter_dimensions"][self.bn_mean]
            yield "bn.var", self.bn_var, self.__model_info["parameter_dimensions"][self.bn_var]
            
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_DC != 0:
            yield "weight", self.dc_g, self.__model_info["parameter_dimensions"][self.w]
            yield "weight", self.dc_h, self.__model_info["parameter_dimensions"][self.w]
            
        else:
            yield "weight", self.w, self.__model_info["parameter_dimensions"][self.w]
            
    def topology_elem_generator(self, compressed_parameter_types_dict):
        """
        Generator method that yields topology elements based on the compressed parameter types.

        Args:
        - compressed_parameter_types_dict (dict): Dictionary containing compressed parameter types.

        Yields:
        - str: Topology element name.
        """
        compressed_parameter_types = compressed_parameter_types_dict[self.block_id]
        
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_DC != 0:
            yield self.dc_g
            yield self.dc_h
            
        else:
            yield self.w
            
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_LS != 0: 
            yield  self.ls
            
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_BN != 0:
            yield self.bn_beta
            yield self.bn_gamma
            yield self.bn_mean
            yield self.bn_var
            
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_BI != 0:
            yield self.bi
        

class NNRModelAccess():
    """
    This class provides an interface for accessing models. 
    It constructs a list of blocks and parameters in the model, 
    and includes a method for yielding instances of NNRBlockAccess or NNRParamAccess for each block or parameter in the model.
    """
    def __init__(self, model_info):
        """
        Initialize the NNRModelAccess object with the given model_info.

        Args:
        - model_info (dict): Model information dictionary containing parameter types, dimensions, and block identifiers.
        """
        self.__model_info = model_info
        self.__block_list = []
        block_set_check = set( model_info["block_identifier"].values() )
        params_sorted = sorted(model_info["parameter_index"], key=model_info["parameter_index"].get)
        
        # Create a list of blocks and individual parameters from the model_info dictionary
        for param in params_sorted:
            if param in model_info["block_identifier"]:
                if model_info["parameter_type"][param] in ["weight"]:
                    self.__block_list.append( (model_info["block_identifier"][param], param) )
                    block_set_check.remove( model_info["block_identifier"][param] )
            else:
                self.__block_list.append( (None, param) )
                
        assert not block_set_check, "Unresolved block identifiers: {}".format( block_set_check )

    def blocks_and_params(self):
        """
        Generator method that yields instances of NNRBlockAccess or NNRParamAccess for each block or parameter in the model.

        Yields:
        - NNRBlockAccess or NNRParamAccess: Object representing a block or parameter in the model.
        """
        for block_id, param in self.__block_list:
            if block_id is None:
                yield NNRParamAccess(self.__model_info, param)
            else:
                yield NNRBlockAccess(self.__model_info, block_id)

        

def set_block_id_and_param_type(model_info, block_id_and_param_type):
    """
    Sets block identifiers and parameter types for a model based on the provided dictionary.

    Args:
    - model_info (dict): The model information dictionary.
    - block_id_and_param_type (dict): Dictionary containing block identifiers and parameter types.

    Raises:
    - AssertionError: If 'block_identifier' or 'parameter_type' is not available in block_id_and_param_type.

    Updates:
    - model_info (dict): Updates the 'block_identifier' field in the model_info dictionary.
    """
    
    # Ensure the necessary fields are in the block_id_and_param_type dictionary
    assert "block_identifier" in block_id_and_param_type, "block_identifier not available!"
    assert "parameter_type" in block_id_and_param_type, "parameter_type not available!"
    
    # Initialize the block_identifier field in model_info
    model_info["block_identifier"] = {}
    
    # Get the list of block identifier values from the block_id_and_param_type dictionary
    block_id_values_list = list(block_id_and_param_type["block_identifier"].values())
    
    # Loop over each parameter and its index in the model_info dictionary
    for param, pardIdx in model_info["parameter_index"].items():
        model_info["parameter_index"][param] = pardIdx

        # Set the parameter type in model_info if it exists in block_id_and_param_type
        if param in block_id_and_param_type["parameter_type"].keys():
            model_info["parameter_type"][param] = block_id_and_param_type["parameter_type"][param]
            
        # Set the block identifier in model_info if it exists in block_id_and_param_type
        if param in block_id_and_param_type["block_identifier"].keys() and block_id_and_param_type["block_identifier"][param] is not None and block_id_values_list.count(block_id_and_param_type["block_identifier"][param]) > 1 :
            model_info["block_identifier"][param] = block_id_and_param_type["block_identifier"][param]
            
            
def add_lsa_to_block_id_and_param_type( block_id_and_param_type, lsa_params ):
    """
    Adds local scaling adaptation (LSA) parameters to a block ID and parameter type dictionary.

    Args:
    - block_id_and_param_type (dict): Dictionary containing block identifiers and parameter types.
    - lsa_params (dict): Dictionary of local scaling adaptation parameters to be added.

    Updates:
    - block_id_and_param_type (dict): Updates the 'block_identifier' and 'parameter_type' fields.

    """
    
    # Loop over each key in the lsa_params dictionary
    for key in lsa_params.keys():
        # If the key is not already in the block_id_and_param_type dictionary, add it
        if key not in block_id_and_param_type["block_identifier"]:
            block_id_and_param_type["block_identifier"][key] = block_id_and_param_type["block_identifier"].get(key.strip("_scaling"), None)
            block_id_and_param_type["parameter_type"][key] = "weight.ls"
            

def sanity_check_block_id_and_param_type(block_id_and_param_type, model_parameters=None):
    """
    Performs a sanity check on a block ID and parameter type dictionary,
    checking the consistency of parameter types and shapes within each block.

    Args:
    - block_id_and_param_type (dict): Dictionary containing block identifiers and parameter types.
    - model_parameters (dict): Optional dictionary of model parameters for shape consistency check.

    Returns:
    - bool: True if the sanity check passes, False otherwise.
    """
    
    # Initialize a dictionary to hold block information
    block_dict = dict()
    
    # Assume the sanity check will pass until proven otherwise
    sanity_check_success = True
    
    # Loop over each parameter and block ID in the block_id_and_param_type dictionary
    for param, blkId in block_id_and_param_type["block_identifier"].items():

        # If the block ID is not None, perform checks
        if blkId != None:
            parT = block_id_and_param_type["parameter_type"][param]
            parShape = model_parameters[param].shape if model_parameters else None
            
            # If the parameter type is not "weight" and the shape is not 1, fail the sanity check
            if model_parameters and parT != "weight" and len(parShape) != 1:
                sanity_check_success = False
                break
            
            # If the block ID is not in block_dict, add it
            if blkId not in block_dict.keys():
                block_dict[blkId] = []
                
            # Add the parameter, its type, and its shape to the block in block_dict
            block_dict[blkId].append([param, parT, parShape])

    # Loop over each block ID and block list in block_dict
    for bId, bList in block_dict.items():
        
        # Define the available types
        available_types = ["weight", "weight.ls", "bias", "bn.mean", "bn.var", "bn.gamma", "bn.beta"]
        lastShape = None

        # Loop over each parameter, its type, and its shape in the block list
        for par, parT, parShape in bList:
            
            if parT not in available_types and parT != "unspecified":
                sanity_check_success = False
                break
            
            # If the parameter type is not "unspecified", remove it from the available types
            if parT != "unspecified":
                available_types.remove(parT) 

            # If the last shape is not None and doesn't match the current shape, fail the sanity check
            if lastShape != None and lastShape[0] != parShape[0]:
                sanity_check_success = False
                break
            
            # Set the last shape to the current shape
            lastShape = parShape

        # If "weight" is still in the available types, fail the sanity check
        if "weight" in available_types:
            sanity_check_success = False
            break
        
    # Return the result of the sanity check
    return sanity_check_success