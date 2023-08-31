from framework.use_case_init import use_cases
from framework.applications.utils import evaluation, transforms
from collections import OrderedDict
import copy, logging
import os
import numpy as np
import nnc_core
import torch

LOGGER = logging.getLogger()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def device_checker(vars_dict):
    
    device_set = set()
    
    for name, variable in vars_dict.items():
        
        if not isinstance(variable, torch.Tensor):
            print(f"Variable {name} is type {type(variable)}")
            continue
            
        device_set.add(variable.device)
        print(f'Tensor {name} is on {variable.device}')

    if len(device_set) > 1:
        print(f"At least two devices are founded : {device_set}")
    else:
        print("All tensors are on the same device.")

# Function to check if a given model object is a PyTorch model
def is_pyt_model( model_object ):
    return isinstance( model_object, torch.nn.Module )

# Function to initialize data functions
def __initialize_data_functions(
                 handler=None,
                 dataset_path=None,
                 batch_size=None,
                 num_workers=None,
                ):
    
    """
    init_~ : load dataset, and prepare dataloader. that's all
    """
    if dataset_path: 
        test_set, test_loader = handler.init_test(
            dataset_path,
            batch_size,
            num_workers
        )

        val_set, val_loader = handler.init_validation(
            dataset_path,
            batch_size,
            num_workers
        )

        train_loader = handler.init_training(
            dataset_path,
            batch_size,
            num_workers
        )
        
    else:
        test_set, test_loader = None, None
        val_set, val_loader   = None, None
        train_loader = None
    
    return test_set, test_loader, val_set, val_loader, train_loader

def create_NNC_model_instance_from_file(
                 model_path,
                 dataset_path=None,
                 lr=1e-4,
                 epochs=30,
                 max_batches=None,
                 batch_size=64,
                 num_workers=1,
                 model_struct=None,
                 lsa=False
                ):
    """Create NNCodec's class instance from model path (and somtimes dataset_path if lsa enabled)

    Args:
        pass

    Returns:
        PYTModel : instance of PytorchModel
        PYTModelExecuter : Model Executer
        model_parameters : state_dict of model
    """

    PYTModel = PytorchModel()
    model_parameters, loaded_model_struct = PYTModel.load_model(model_path)
    if model_struct == None and loaded_model_struct != None:
        model_struct = loaded_model_struct

    if dataset_path and model_struct:
        PYTModelExecuter = create_imagenet_model_executer(model_struct=model_struct,
                                                          dataset_path=dataset_path,
                                                          lr=lr,
                                                          epochs=epochs,
                                                          max_batches=max_batches,
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          lsa=lsa
                                                          )
        if lsa:
            model_parameters = PYTModel.init_model_from_dict(PYTModelExecuter.model.state_dict())
    else:
        PYTModelExecuter = None

    return PYTModel, PYTModelExecuter, model_parameters


########################### CUSTOM MODIFIED ############################
"""
Add funtionality to check task_type and initialize PYTModelExecuter Adaptively
Using create_imagenet_model_executer or create_blender_model_executer
"""
def create_NNC_model_instance_from_object(
                 model_object,
                 dataset_path,
                 lsa,
                 lr,
                 epochs,
                 task_type,
                 dataset_type,
                 N_iters,
                 learning_rate_decay,
                 i_save,
                 max_batches=None,
                 batch_size=64, # Only Applied for Classification Task
                 num_workers=1,
                 model_struct=None
                 ):

    PYTModel = PytorchModel()
    
    # Returns parameter of PYTModel, and model object (loaded_model_struct == model_object)
    # and internally, it changes private member variable PYTModel.__model_info
    model_parameters, loaded_model_struct = PYTModel.init_model_from_model_object(model_object)
    
    if model_struct == None and loaded_model_struct != None:
        model_struct = loaded_model_struct

    if dataset_path and model_struct:
        
        # Classification : Official supported task (ImageNet + Classification)
        if task_type == 'Classification':
            PYTModelExecuter = create_imagenet_model_executer(model_struct=model_struct,
                                                            dataset_path=dataset_path,
                                                            lr=lr,
                                                            epochs=epochs,
                                                            max_batches=max_batches,
                                                            batch_size=batch_size,
                                                            num_workers=num_workers,
                                                            lsa=lsa
                                                            )
        # Custom added task type
        elif task_type == 'NeRF':
            
            assert dataset_type in ['blender', 'llff']
            
            PYTModelExecuter = create_nerf_model_executer( model_struct=model_struct,
                                                           dataset_type=dataset_type,
                                                           lr=lr,
                                                           epochs=epochs,
                                                           max_batches=max_batches,
                                                           lsa=lsa,
                                                           N_iters = N_iters, 
                                                           learning_rate_decay = learning_rate_decay, 
                                                           i_save = i_save
                                                           )
        else:
            # If task_type is neither 'Classification' nor 'NeRF', raise an error
            raise ValueError("Invalid task_type. Supported values are 'Classification' and 'NeRF'.")
        
        if lsa:
            # if LSA enabled
            # Model parameters have to include additional LSA parameter
            # So, model_parameters should changed to lsa-added ones
            model_parameters = PYTModel.init_model_from_dict(PYTModelExecuter.model.state_dict())
            
    else:
        PYTModelExecuter = None

    return PYTModel, PYTModelExecuter, model_parameters
###################################################################################

def create_imagenet_model_executer( 
                            model_struct,
                            dataset_path,
                            lr=1e-4,
                            epochs=30,
                            max_batches=None,
                            batch_size=64,
                            num_workers=1,
                            lsa=False,
                            ):
    """
    1. __initialize_data_functions : initialize train/val/test loader and dataset
    2. Executes Model using ImageNetPytorchModelExecuter
    """
    
    assert model_struct != None, "model_struct must be specified in order to create a model_executer!"
    assert dataset_path != None, "dataset_path must be specified in order to create a model_executer!"
    
    handler = use_cases['NNR_PYT']
    
    # __initialize_data_functions : initialize train/val/test loader and dataset
    test_set, test_loader, val_set, val_loader, train_loader = __initialize_data_functions( handler=handler,
                                                                              dataset_path=dataset_path,
                                                                              batch_size=batch_size,
                                                                              num_workers=num_workers)

    assert (test_set!=None and test_loader!= None) or ( val_set!= None and val_loader!= None ), \
        "Any of the pairs test_set/test_loader or val_set/val_loader must be specified in order to use data driven optimizations methods!"
    
    # Main Part of This Function
    PYTModelExecuter = ImageNetPytorchModelExecuter(handler,
                                                    train_loader=train_loader,
                                                    test_loader=test_loader,
                                                    test_set=test_set,
                                                    val_loader=val_loader,
                                                    val_set=val_set,
                                                    model_struct=model_struct,
                                                    lsa=lsa,
                                                    lr=lr,
                                                    epochs=epochs,
                                                    max_batches=max_batches)

    PYTModelExecuter.initialize_optimizer(lr=lr)

    return PYTModelExecuter


def save_to_pytorch_file( model_data, path ):
    model_dict = OrderedDict()
    for module_name in model_data:
        model_dict[module_name] = torch.tensor(model_data[module_name])
    torch.save(model_dict, path)


def get_model_file_with_parameters( parameters, model_struct ):

    new_model_struct = copy.deepcopy(model_struct)

    state_dict = OrderedDict()
    for param in parameters.keys():
        state_dict[param] = torch.tensor( parameters[param] )
        assert param in new_model_struct.state_dict(), "The provided model_strcut does not fit the parameter state dict decoded from the bitstream! Parameter '{}' not found in model_struct state dict!".format(param)

    new_model_struct.load_state_dict(state_dict)

    return new_model_struct


class PytorchModel(nnc_core.nnr_model.NNRModel):
    
    def __init__(self, model_dict=None):
        
        if model_dict and isinstance(model_dict, dict):
            self.init_model_from_dict(model_dict)
        else:
            self.__model_info = None

        self.device = DEVICE
    
    def load_model(self, 
                   model_path
                  ):
        """
        "Robustly Load Model"
        Simply load model from model_path
        Can include from OrderedDict, dict, nn.Module(whole model), and even multi-GPU trained model
        Returns the PYTModel instance and parameters dict
        """
        
        model_file = torch.load(model_path, map_location=self.device) ##loads the state_dict
        model_struct = None
        
        try:
            model_parameter_dict = None

            # model state_dict
            if isinstance(model_file, OrderedDict):
                model_parameter_dict = model_file

            # checkpoint including state_dict
            elif isinstance(model_file, dict):
                for key in model_file.keys():
                    if isinstance(model_file[key], OrderedDict):
                        model_parameter_dict = model_file[key]
                        print("Loaded weights from state_dict '{}' from checkpoint elements {}".format(key,
                                                                                                       model_file.keys()))
                        break
                if not model_parameter_dict:
                    assert 0, "Checkpoint does not include a state_dict in {}".format(model_file.keys())

            # whole model (in general not recommended)
            elif isinstance(model_file, torch.nn.Module):
                model_parameter_dict = model_file.state_dict()
                model_struct = model_file

            # multi-GPU parallel trained models (torch.nn.DataParallel)
            if all(i[:7] == 'module.' for i in model_parameter_dict.keys()):
                print("Removing 'module.' prefixes from state_dict keys resulting from saving torch.nn.DataParallel "
                      "models not in the recommended way, that is torch.save(model.module.state_dict()")
                new_state_dict = OrderedDict()
                for n, t in model_parameter_dict.items():
                    name = n[7:]  # remove `module.`
                    new_state_dict[name] = t
                model_parameter_dict = new_state_dict

        except:
            raise SystemExit("Can't read model: {}".format(model_path))
        
        # intializes the model and the state dict and returns the PYTModel instance and parameters dict
        # model_parameter_dict : OrderedDict Type / model_struct : input itself (Only when input is nn.Module type)
        return self.init_model_from_dict(model_parameter_dict), model_struct 


    def init_model_from_model_object(   self,
                                        model_object,
                                    ):

        return self.init_model_from_dict(model_object.state_dict()), model_object


    @property
    def model_info(self):
        return self.__model_info

    def init_model_from_dict(self, pt_dict):

        # Step 1: Check the type of pt_dict and assign model_dict accordingly
        if isinstance(pt_dict, dict):
            model_dict = pt_dict
        elif isinstance(pt_dict.state_dict(), dict):
            model_dict = pt_dict.state_dict()

        # Step 2: Initialize dictionaries to store model information and parameters
        model_data = {
            'parameters': {}, 
            'reduction_method': 'baseline'
            }
        
        model_info = {
            'parameter_type': {}, 
            'parameter_dimensions': {}, 
            'parameter_index': {}, 
            'block_identifier': {}, 
            'original_size': {},
            'topology_storage_format': nnc_core.nnr_model.TopologyStorageFormat.NNR_TPL_PYT,
            'topology_compression_format': nnc_core.nnr_model.TopologyCompressionFormat.NNR_PT_RAW
            }
        
        """ Expected Return of model_data, model_info as example
        
        model_data = {
            'parameters' : {
                'conv1.weight': np.array([[[[...]], [[...]], ..., [[...]]]),  # Array of shape (16, 3, 3, 3)
                'bn1.weight': np.array([...]),  # Array of shape (16,)
                'bn1.bias': np.array([...]),  # Array of shape (16,)
                'fc.weight': np.array([[...], [...], ..., [...]]),  # Array of shape (10, 256)
                'fc.bias': np.array([...])  # Array of shape (10,)
            }

        }
        
        model_info = {
            'parameter_type': {
                'conv1.weight': 'weight',
                'bn1.weight': 'weight',
                'bn1.bias': 'bias',
                'fc.weight': 'weight',
                'fc.bias': 'bias'
            },
            'parameter_dimensions': {
                'conv1.weight': (16, 3, 3, 3),
                'bn1.weight': (16,),
                'bn1.bias': (16,),
                'fc.weight': (10, 256),
                'fc.bias': (10,)
            },
            'parameter_index': {
                'conv1.weight': 0,
                'bn1.weight': 1,
                'bn1.bias': 2,
                'fc.weight': 3,
                'fc.bias': 4
            },
            'block_identifier': {},
            'original_size': 17650,
            'topology_storage_format': nnc_core.nnr_model.TopologyStorageFormat.NNR_TPL_PYT,
            'topology_compression_format': nnc_core.nnr_model.TopologyCompressionFormat.NNR_PT_RAW
        }
        
        """

        # metadata only needed for MNASNet from PYT model zoo... further work: include into bitstream
        # self.metadata = getattr(model_dict, '_metadata', None)

        # Step 3: Prepare some lists for type comparison
        type_list_int = ['int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32']
        type_list_1_bytes = ['int8', 'uint8']
        type_list_2_bytes = ['int16', 'uint16', 'float16']
        original_size = 0
        
        # Step 4: Iterate over each module in the model_dict
        # model_dict = state_dict of model
        for i, module_name in enumerate(model_dict): # Iterate each module
            
            # Skip '.num_batches_tracked' module as it's not relevant
            if '.num_batches_tracked' in module_name:
                continue
                
            # Determine the data type and calculate the original size
            # model_dict[module_name].data : weight parameter of model
            if model_dict[module_name].data.cpu().detach().numpy().dtype in type_list_1_bytes: # int8 & uint8
                original_size += model_dict[module_name].numel() # numel : Number of Elements
            elif model_dict[module_name].data.cpu().detach().numpy().dtype in type_list_2_bytes: # int16 & uint16 & float16
                original_size += model_dict[module_name].numel()*2
            else:
                original_size += model_dict[module_name].numel()*4
            
            # Store the tensor values in model_data['parameters'] dictionary
            model_data['parameters'][module_name] = np.int32(model_dict[module_name].data.cpu().detach().numpy()) if model_dict[module_name].data.cpu().detach().numpy().dtype in type_list_int else model_dict[module_name].data.cpu().detach().numpy()
            
            # Flatten tensor values for modules with '.weight_scaling' in the name
            # weight_scaling : parameter to do LSA
            if '.weight_scaling' in module_name:
                model_data['parameters'][module_name] = model_data['parameters'][module_name].flatten()
                
            # Store the shape of the tensor in model_info['parameter_dimensions'] dictionary
            mdl_shape = model_data['parameters'][module_name].shape
            model_info['parameter_dimensions'][module_name] = mdl_shape
            
            # If the tensor is a scalar, convert it to a float32 array with a single element
            if len(mdl_shape) == 0:  # scalar
                model_data['parameters'][module_name] = np.array([np.float32(model_data['parameters'][module_name])])
                model_info['parameter_dimensions'][module_name] = np.array([0]).shape
                
            # Store the index of the module in model_info['parameter_index'] dictionary
            model_info['parameter_index'][module_name] = i

            # Determine the type of the parameter based on its name and dimensions
            dims = len(mdl_shape)

            if dims > 1 and '.weight' in module_name:
                model_info['parameter_type'][module_name] = 'weight'
            elif dims > 1:
                model_info['parameter_type'][module_name] = 'weight'
            elif dims == 1:
                if '.bias' in module_name:
                    model_info['parameter_type'][module_name] = 'bias' 
                elif '.running_mean' in module_name:
                    model_info['parameter_type'][module_name] = 'bn.mean'
                elif '.running_var' in module_name:
                    model_info['parameter_type'][module_name] = 'bn.var'
                elif '.weight_scaling' in module_name:
                    model_info['parameter_type'][module_name] = 'weight.ls'
                elif 'gamma' in module_name:
                    model_info['parameter_type'][module_name] = 'bn.gamma'
                elif '.weight' in module_name:
                    model_info['parameter_type'][module_name] = "weight"
                else:
                    model_info['parameter_type'][module_name] = 'unspecified'
            else:
                model_info['parameter_type'][module_name] = 'unspecified'
            
        # Store the calculated original size in model_info['original_size']
        model_info["original_size"] = original_size

        # Set model_info as an attribute of the class
        self.__model_info = model_info

        # Return the processed model parameters
        # it seems to be only returns model_data weight, but internally stores model info (__model_info)
        return model_data["parameters"] 

    def save_state_dict(self, path, model_data):
    
        model_dict = OrderedDict()
        
        for module_name in model_data:
            model_dict[module_name] = torch.tensor(model_data[module_name])
            
            if model_data[module_name].size == 1:
                model_dict[module_name] = torch.tensor(np.int64(model_data[module_name][0]))
                
        torch.save(model_dict, path)
    
    def guess_block_id_and_param_type(self, model_parameters):
        
        try:
            # Initialize dictionaries to store inferred block identifiers and parameter types
            block_id_and_param_type = {"block_identifier" : {}, "parameter_type" : {}}
            block_dict = dict() # Dictionary to store information about different blocks found in the model
            blkNum = -1 # Counter to generate unique block identifiers
            
            # Iterate through each parameter in the model_parameters dictionary
            for param in model_parameters.keys():
                dims = len(model_parameters[param].shape) # Determine the dimensions of the parameter's tensor
                paramShape = model_parameters[param].shape # Get the shape of the parameter's tensor
                splitted_param = param.split(".")
                param_end = splitted_param[-1] # Get the last part of the parameter name
                base_block_id  = ".".join(splitted_param[0:-1]+[""]) if len(splitted_param[0:-1]) != 0 else "genericBlk."

                # Infer the type (paramType) and base block identifier (base_block_id) based on the parameter name and dimensions
                if dims > 1 and ('kernel' in param_end or 'weight' in param_end):
                    paramType = 'weight'
                    blockId = base_block_id
                elif dims > 1:
                    paramType = 'weight'
                    blockId = base_block_id
                elif dims == 1:
                    if 'bias' in param_end or 'beta' in param_end:
                        paramType = 'bias'
                        blockId = base_block_id
                    elif 'running_mean' in param_end or 'moving_mean' in param_end:
                        paramType = 'bn.mean'
                        blockId = base_block_id
                    elif 'running_var' in param_end or 'moving_variance' in param_end:                        
                        paramType = 'bn.var'
                        blockId = base_block_id
                    elif 'weight_scaling' in param_end:
                        paramType = 'weight.ls'
                        blockId = base_block_id
                    elif 'gamma' in param_end:
                        paramType = 'bn.gamma'
                        blockId = base_block_id
                    elif 'weight' in param_end:
                        paramType = 'weight'
                        blockId = base_block_id
                    else:
                        paramType = 'unspecified'
                        blockId = None
                else:
                    paramType = 'unspecified'
                    blockId = None
                
                if blockId:
                    block_id = base_block_id + str(blkNum)
                    if block_id in block_dict.keys():
                        if any([a[1] == paramType for a in block_dict[block_id]]):
                            blkNum += 1
                        block_id = base_block_id + str(blkNum)
                        blockId = base_block_id + str(blkNum)
                    else:
                        blkNum += 1
                        block_id = base_block_id + str(blkNum)
                        blockId = base_block_id + str(blkNum)
                            
                    if block_id not in block_dict.keys():
                        block_dict[block_id] = []
                        
                    block_dict[block_id].append( [param, paramType, blockId, dims, paramShape] )
                else:
                    block_id_and_param_type["parameter_type"][param] = paramType
                    block_id_and_param_type["block_identifier"][param] = blockId
                            
            weight_block_list = []
            bn_block_list     = []

            for block_list in block_dict.values():
                if any(["bn." in a[1] for a in block_list]):
                
                    for i, val in enumerate(block_list):
                        par, parT, blkId, dims, _ = val
                        if parT == 'weight' and dims == 1:
                            block_list[i][1] = "bn.gamma"
                        if parT == 'bias':
                            block_list[i][1] = "bn.beta"
                            
                    bn_block_list.append( block_list )
                else:
                    weight_block_list.append(block_list)
                    
            weight_shape = None
            weight_blkId = None
            
            for weight_block in weight_block_list:
                weight_shape = None
                weight_blkId = None
                
                for par, parT, blkId, dims, paramSh in weight_block:
                    block_id_and_param_type["parameter_type"][par] = parT
                    block_id_and_param_type["block_identifier"][par] = blkId
                    
                    if parT == 'weight':
                        weight_shape = paramSh
                        weight_blkId = blkId
            
                if len(bn_block_list) != 0 and any([dim == bn_block_list[0][0][4][0] for dim in weight_shape]):
                    bn_block = bn_block_list.pop(0)
                    
                    for par, parT, _, _, _ in bn_block:
                        block_id_and_param_type["parameter_type"][par] = parT
                        block_id_and_param_type["block_identifier"][par] = weight_blkId
            
            assert len(bn_block_list) == 0
                                
        except:
            print("INFO: Guessing of block_id_and_parameter_type failed! block_id_and_parameter_type has been set to 'None'!")
            block_id_and_param_type = None
                
        return block_id_and_param_type


class ImageNetPytorchModelExecuter(nnc_core.nnr_model.ModelExecute):
    
    """
    Key Function of Making "ModelExecuter" when using PyTorch model & ImageNet
    """

    def __init__(self, 
                 handler,
                 train_loader,
                 test_loader,
                 test_set,
                 val_loader,
                 val_set,
                 model_struct,
                 lsa,
                 max_batches=None,
                 epochs=5,
                 lr=1e-4,
                 ):

        self.device = DEVICE
        
        torch.manual_seed(451)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.learning_rate = lr
        self.epochs = epochs
        self.max_batches = max_batches
        self.handle = handler
        
        ############## set dataset and dataloader ##############
        if test_set:
            self.test_set = test_set
            self.test_loader = test_loader
        if val_set:
            self.val_set = val_set
            self.val_loader = val_loader
        if train_loader:
            self.train_loader = train_loader
        ########################################################
        
        if model_struct:
            self.original_model = copy.deepcopy(model_struct)
            
            if lsa:
                lsa_gen = transforms.LSA(model_struct) # create instance of LSA class
                model_struct = lsa_gen.add_lsa_params() # add_lsa_params : adds LSA scaling parameters to conv and linear layers
                
            self.model = model_struct
        else:
            self.original_model = None
            self.model = None

    def test_model(self,
                   parameters,
                   verbose=False
                   ):

        torch.set_num_threads(1)
        Model = copy.deepcopy(self.model)

        base_model_arch = Model.state_dict()
        model_dict = OrderedDict()
        
        # Copy state_dict of model into new OrderedDict type "model_dict"
        # With regarding of weight_scaling parameter, which is key param of LSA
        for module_name in base_model_arch:
            
            if module_name in parameters:    
                # Copy the parameters from prarameters to model_dict
                # When weight tensor is not tensor, convert it to match
                model_dict[module_name] = parameters[module_name] if torch.is_tensor(parameters[module_name]) else \
                    torch.tensor(parameters[module_name])
                    
                if "weight_scaling" in module_name:
                    # Reshape the weight scaling parameters to match the model architecture
                   model_dict[module_name] = model_dict[module_name].reshape(base_model_arch[module_name].shape)

        for param in Model.state_dict().keys():
            if "num_batches_tracked" in param:
                continue # skip
            assert param in parameters.keys(), \
                "The provided model_struct does not fit the parameter state dict decoded from the bitstream! Parameter '{}' not found in parameter dict!".format(param)
        
        Model.load_state_dict(model_dict)
        
        """ Handler : use_case["NNR_PYT"]
        
            "NNR_PYT":  ModelSetting( None,
                            evaluation.evaluate_classification_model,
                            train.train_classification_model,
                            imagenet.imagenet_dataloaders,
                            torch.nn.CrossEntropyLoss()
                            ),
            
            --> evaluate : evaluate performance on ImangeNet
        """
        accs = self.handle.evaluate(
            Model,
            self.handle.criterion,
            self.test_loader,
            self.test_set,
            device=self.device,
            verbose=verbose
        )
        del Model
        return accs
    
    # Same as test_model
    def eval_model(self,
                   parameters,
                   verbose=False
                   ):

        torch.set_num_threads(1)

        Model = copy.deepcopy(self.model)

        base_model_arch = Model.state_dict()
        model_dict = OrderedDict()
        
        for module_name in base_model_arch:
            
            if module_name in parameters:
                model_dict[module_name] = parameters[module_name] if torch.is_tensor(parameters[module_name]) else \
                    torch.tensor(parameters[module_name])
                    
                if "weight_scaling" in module_name:
                   model_dict[module_name] = model_dict[module_name].reshape(base_model_arch[module_name].shape)

        Model.load_state_dict(model_dict)

        accs = self.handle.evaluate(
            Model,
            self.handle.criterion,
            self.val_loader,
            self.val_set,
            device=self.device,
            verbose=verbose
        )

        del Model
        return accs

    def tune_model(
            self,
            parameters,
            param_types,
            lsa_flag=False,
            ft_flag=False,
            verbose=False,     
    ):
        
        # Set the number of threads for PyTorch
        torch.set_num_threads(1)
        verbose = 1 if (verbose & 1) else 0

        # Create a copy of the model's state_dict in a new OrderedDict
        """ base_model_arch : Represents Initial State of self.model before fine_tuning """
        base_model_arch = self.model.state_dict() # use model_architecture as state_dict()
        model_dict = OrderedDict()
        
        # This Iteration is Same As test_model(), and eval_model()
        for module_name in base_model_arch:
            
            # Copy the parameters from parameters to model_dict
            # When weight tensor is not a tensor, convert it to match
            if module_name in parameters:
                
                model_dict[module_name] = parameters[module_name] if torch.is_tensor(parameters[module_name]) else \
                    torch.tensor(parameters[module_name])
                    
                if "weight_scaling" in module_name:
                    # Reshape the weight scaling parameters to match the model architecture
                   model_dict[module_name] = model_dict[module_name].reshape(base_model_arch[module_name].shape)
        
        # Load the copied model_dict into the model to update its parameters
        self.model.load_state_dict(model_dict)

        for param in parameters:
            parameters[param] = copy.deepcopy(self.model.state_dict()[param])

        tuning_params = []
        
        """ param_types may look like : 
        
                param_types = {
                    'module_name1.weight': 'weight.ls',  # Parameter name and its type
                    'module_name1.bias': 'bias',         # Another parameter name and its type
                    'module_name2.weight': 'weight.conv',
                    'module_name2.bias': 'bias',
                    # Add more parameter names and their corresponding types here
                }
        """

        for name, param in self.model.named_parameters():
            # O_TYPES = ["weight.ls","bias","bn.beta","bn.gamma","bn.mean","bn.var","unspecified"]
            if lsa_flag and ft_flag and param_types[name] in nnc_core.nnr_model.O_TYPES:
                param.requires_grad = True
                tuning_params.append(param)
            elif lsa_flag and param_types[name] == 'weight.ls':
                param.requires_grad = True
                tuning_params.append(param)
            elif ft_flag and param_types[name] != 'weight.ls' and param_types[name] in nnc_core.nnr_model.O_TYPES:
                param.requires_grad = True
                tuning_params.append(param)
            else:
                param.requires_grad = False
        #
        #################### May Related to LSA ########################
        # optimizer to tune tuning_params, which is LSA!... but, where to train?
        self.tuning_optimizer = torch.optim.Adam(tuning_params, lr=self.learning_rate)
        ################################################################
        
        ############ Evaluate Model Performance Before LSA ############
        # perf : tuple -> (top1, top5, loss)
        perf = self.eval_model(parameters, verbose=verbose)
        
        best_loss, best_params = perf[2], copy.deepcopy(parameters)
        
        if verbose:
            print(f'Validation accuracy (loss) before LSA and/or Fine Tuning: {perf[0]} ({perf[2]})')
            print(f'Test performance (top1, top5, loss) before LSA and/or Fine Tuning: '
                  f'{self.test_model(parameters, verbose=verbose)}')
        ###############################################################
        
        ############# Training of Model With LSA ##############
        # self.handle.train : see framework/use_case_init
        for e in range(self.epochs):
            train_acc, loss = self.handle.train(
                nerf_wrapper = self.model,
                optimizer = self.tuning_optimizer, # optimizer to tune tuning_params (LSA-related maybe)
                device=self.device,
                freeze_batch_norm=True if lsa_flag and not ft_flag else False,
            )
        ######################################################
            
            print(f'Epoch {e+1}: Train accuracy: {train_acc}, Loss: {loss}')
            
            for param in parameters:
                parameters[param] = copy.deepcopy(self.model.state_dict()[param])

        ############ Evaluate Model Performance After LSA ############
            perf = self.eval_model(parameters, verbose=verbose)
            
            if perf[2] < best_loss and best_loss - perf[2] > 1e-3:
                best_loss = perf[2]
                best_params = copy.deepcopy(parameters)
            else:
                if verbose:
                    print(f'Early Stopping due to model convergence or overfitting')
                    print(f'Epoch {e + 1}: Validation accuracy (loss) after Model Tuning: {perf[0]} ({perf[2]})')
                break
            
            if verbose:
                if lsa_flag and not ft_flag:
                    print(f'Epoch {e+1}: Validation accuracy (loss) after Model Tuning: {perf[0]} ({perf[2]})')
        if verbose:
            print(f'Test performance (top1, top5, loss) after LSA and/or Fine Tuning: '
                  f'{self.test_model(parameters, verbose=verbose)}')
        ##############################################################
        
        lsa_params, ft_params = {}, {}
        
        for name in best_params:
            if lsa_flag and param_types[name] == 'weight.ls':
                lsa_params[name] = best_params[name].cpu().numpy().flatten()
            elif ft_flag and param_types[name] != 'weight.ls' and param_types[name] in nnc_core.nnr_model.O_TYPES:
                ft_params[name] = best_params[name].cpu().numpy()
                
        return (lsa_params, ft_params)

    def initialize_optimizer(self,
                             lr=1e-5,
                             mdl_params=None
                             ):
        if mdl_params:
            mdl_params_copy = copy.deepcopy(mdl_params)
            Model = self.model
            base_model_arch = Model.state_dict()
            model_dict = OrderedDict()
            
            for module_name in base_model_arch:
                if module_name in mdl_params_copy:
                    model_dict[module_name] = torch.tensor(mdl_params_copy[module_name])
                else:
                    model_dict[module_name] = base_model_arch[module_name]
                    
            Model.load_state_dict(model_dict)

        if hasattr(self, "model"):
            params = [param for name, param in self.model.named_parameters()
                        if not '.weight_scaling' in name]
            self.optimizer = torch.optim.Adam(params, lr=lr)

    def has_eval(self):
        return True
    
    def has_test(self):
        return True
    
    def has_tune_ft(self):
        return True
    
    def has_tune_lsa(self):
        return True
    
    
########################### CUSTOM ADDED #############################

def create_nerf_model_executer(model_struct,
                        dataset_type,
                        lr,
                        epochs,
                        lsa,
                        N_iters, 
                        learning_rate_decay, 
                        i_save,
                        max_batches = False
                        ):
    """
    1. __initialize_data_functions : initialize train/val/test loader and dataset
    2. Executes Model using ImageNetPytorchModelExecuter <-- input as train/val/test loader
    """
    
    handler = use_cases['NERF_PYT']
    
    # Main Part of This Function
    # Use Dummy DataLoader First
    
    nerf_modelexecuter = NeRFModelExecuter( handler = handler,
                                            model_struct = model_struct,
                                            dataset_type = dataset_type,
                                            lsa = lsa,
                                            lr = lr,
                                            epochs = epochs,
                                            max_batches = max_batches,
                                            N_iters = N_iters, 
                                            learning_rate_decay = learning_rate_decay, 
                                            i_save = i_save
                                            )
    

    # nerf_modelexecuter.initialize_optimizer(lr=lr)

    return nerf_modelexecuter
    
class NeRFModelExecuter(nnc_core.nnr_model.ModelExecute):

    def __init__(self, 
                 handler, # train, val, test dataset & dataloader
                 model_struct,
                 dataset_type : str, 
                 lsa : bool,
                 lr : float,
                 epochs : int,
                 max_batches : int,
                 N_iters : int, 
                 learning_rate_decay : float, 
                 i_save : int
                 ):
        
        self.device = DEVICE
        
        torch.manual_seed(451)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.dataset_type = dataset_type
        self.learning_rate = lr
        self.epochs = epochs
        self.max_batches = max_batches
        self.handle = handler
        self.N_iters : int = N_iters
        self.learning_rate_decay = learning_rate_decay
        self.i_save = i_save
        
        
        if model_struct:
            self.original_model = copy.deepcopy(model_struct)
            
            if lsa:
                lsa_gen = transforms.LSA(model_struct) # create instance of LSA class
                model_struct = lsa_gen.add_lsa_params() # add_lsa_params : adds LSA scaling parameters linear layers
                
            self.model = model_struct
        else:
            raise ValueError("model_struct is not properly set!")
            
    def test_model(self,
                   parameters,
                   verbose=False
                   ):

        torch.set_num_threads(1)
        Model = copy.deepcopy(self.model)

        base_model_arch = Model.state_dict()
        model_dict = OrderedDict()
        
        # Copy state_dict of model into new OrderedDict type "model_dict"
        # With regarding of weight_scaling parameter, which is key param of LSA
        for module_name in base_model_arch:
            
            if module_name in parameters:    
                # Copy the parameters from prarameters to model_dict
                # When weight tensor is not tensor, convert it to match
                model_dict[module_name] = parameters[module_name] if torch.is_tensor(parameters[module_name]) else \
                    torch.tensor(parameters[module_name])
                    
                if "weight_scaling" in module_name:
                    # Reshape the weight scaling parameters to match the model architecture
                   model_dict[module_name] = model_dict[module_name].reshape(base_model_arch[module_name].shape)

        for param in Model.state_dict().keys():
            if "num_batches_tracked" in param:
                continue # skip
            assert param in parameters.keys(), \
                "The provided model_struct does not fit the parameter state dict decoded from the bitstream! Parameter '{}' not found in parameter dict!".format(param)
        
        Model.load_state_dict(model_dict)
        
        # MOCK VERSION
        acc = self.handle.evaluate(
            model = Model,
            device = self.device
        )
        
        del Model
        return acc
    
    def eval_model(self,
                   parameters,
                   verbose=False
                   ):

        torch.set_num_threads(1)

        Model = copy.deepcopy(self.model)

        base_model_arch = Model.state_dict()
        model_dict = OrderedDict()
        
        for module_name in base_model_arch:
            
            if module_name in parameters:
                model_dict[module_name] = parameters[module_name] if torch.is_tensor(parameters[module_name]) else \
                    torch.tensor(parameters[module_name])
                    
                if "weight_scaling" in module_name:
                   model_dict[module_name] = model_dict[module_name].reshape(base_model_arch[module_name].shape)

        Model.load_state_dict(model_dict)
        
        # MOCK VERSION
        acc = self.handle.evaluate(
            Model,
            device = self.device
        )

        del Model
        return acc

    def tune_model(
            self,
            bitstream_path,
            parameters,
            param_types,
            lsa_flag=True,
            ft_flag=False,
            verbose=False,
    ):
        
        # Set the number of threads for PyTorch
        torch.set_num_threads(1)
        verbose = 1 if (verbose & 1) else 0

        # Create a copy of the model's state_dict in a new OrderedDict
        """ base_model_arch : Represents Initial State of self.model before fine_tuning """
        base_model_arch = self.model.state_dict() # use model_architecture as state_dict()
        model_dict = OrderedDict()
        
        # This Iteration is Same As test_model(), and eval_model()
        for module_name in base_model_arch:
            
            # Copy the parameters from parameters to model_dict
            # When weight tensor is not a tensor, convert it to match
            if module_name in parameters:
                
                model_dict[module_name] = parameters[module_name] if torch.is_tensor(parameters[module_name]) else \
                    torch.tensor(parameters[module_name])
                    
                if "weight_scaling" in module_name:
                    # Reshape the weight scaling parameters to match the model architecture
                    model_dict[module_name] = model_dict[module_name].reshape(base_model_arch[module_name].shape)
        
        # Load the copied model_dict into the model to update its parameters
        self.model.load_state_dict(model_dict)

        for param in parameters: # iterate key of parameter(dict)
            parameters[param] = copy.deepcopy(self.model.state_dict()[param])

        tuning_params = []
        
        """ param_types may look like : 
        
                param_types = {
                    'module_name1.weight': 'weight.ls',  # Parameter name and its type
                    'module_name1.bias': 'bias',         # Another parameter name and its type
                    'module_name2.weight': 'weight.conv',
                    'module_name2.bias': 'bias',
                    # Add more parameter names and their corresponding types here
                }
        """

        for name, param in self.model.named_parameters():
            # print(name)
            
            # O_TYPES = ["weight.ls","bias","bn.beta","bn.gamma","bn.mean","bn.var","unspecified"]
            
            if lsa_flag and ft_flag and param_types[name] in nnc_core.nnr_model.O_TYPES:
                param.requires_grad = True
                tuning_params.append(param)
            elif lsa_flag and param_types[name] == 'weight.ls':
                # <=> name : ~~~~~.weight_scaling
                param.requires_grad = True
                tuning_params.append(param)
            elif ft_flag and param_types[name] != 'weight.ls' and param_types[name] in nnc_core.nnr_model.O_TYPES:
                param.requires_grad = True
                tuning_params.append(param)
            else:
                param.requires_grad = False
        
        print('\n\n')
        print("###########################################################################################")
        print("ENTERING CUSTOM MODE : NERF + LSA")
        print("WARNING! This mode is currently in alpha testing and may not function as expected.")
        print("\nHere are some features of NeRF + LSA Compared to Classification + LSA")
        print("1. Early Stopping Based On Validation Accuracy Is Not Used. Uses Model of Final Epoch")
        print("2. Learning Rate Decay is Applied on Every Epoch (= Every N_iter times of iteration in run_nerf.py)")
        print("3. Precrop Which is Used at the Early Stage of Training - Deprecated")
        print("4. Training Result is Saved in Real-Time")
        print("Starting LSA parameter tuning process...")
        print("###########################################################################################")
        print('\n\n')
            
        
        self.model.global_step = 0
        self.model.tuning_optimizer = torch.optim.Adam(tuning_params, lr=self.learning_rate)
        
        if self.learning_rate_decay != 0:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.model.tuning_optimizer, 
                                                        step_size=1, 
                                                        gamma=self.learning_rate_decay)
        
        for e in range(self.epochs):
            
            train_acc, loss = self.handle.train(
                nerf_wrapper = self.model,
                dataset_type = self.dataset_type,
                freeze_batch_norm = True,
                basedir_save = os.path.dirname(os.path.dirname(bitstream_path)),
                N_iters = self.N_iters,
                i_save = self.i_save
            )
            
            if self.learning_rate_decay != 0: 
                scheduler.step() # Learning rate decay to 0.1 every epoch (Every N_iter times of iterations)
            
            
            print(f'Epoch {e+1} done. Train accuracy: {train_acc}, Loss: {loss}')
            
            """
            IN THE CASE OF NERF MODEL, THERE IS NO PROCESS OF EARLY-STOPPING
            """
            
            for param in parameters:
                parameters[param] = copy.deepcopy(self.model.state_dict()[param])

            best_params = copy.deepcopy(parameters)

        lsa_params, ft_params = {}, {}
        
        for name in best_params:
            if lsa_flag and param_types[name] == 'weight.ls':
                lsa_params[name] = best_params[name].cpu().numpy().flatten()
            elif ft_flag and param_types[name] != 'weight.ls' and param_types[name] in nnc_core.nnr_model.O_TYPES:
                ft_params[name] = best_params[name].cpu().numpy()
        
        return (lsa_params, ft_params)

    def has_eval(self):
        return True
    
    def has_test(self):
        return True
    
    def has_tune_ft(self):
        return True
    
    def has_tune_lsa(self):
        return True
    
###########################################################################