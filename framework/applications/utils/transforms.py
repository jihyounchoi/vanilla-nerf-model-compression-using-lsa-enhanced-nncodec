
import os
import torch
import copy
import torch.nn as nn
from torch.functional import F
import cv2 as cv
import numpy as np


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##############################################################################################

def transforms_tef_model_zoo(filename, label, image_size=224):
    # Read image and convert to float32
    img = cv.imread(filename.numpy().decode()).astype(np.float32)

    resize = 256
    if image_size > 224:
        resize = image_size

    # Resize
    height, width, _ = img.shape
    new_height = height * resize // min(img.shape[:2])
    new_width = width * resize // min(img.shape[:2])
    img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)

    # Crop
    height, width, _ = img.shape
    startx = width // 2 - (image_size // 2)
    starty = height // 2 - (image_size // 2)
    img = img[starty:starty + image_size, startx:startx + image_size]
    assert img.shape[0] == image_size and img.shape[1] == image_size, (img.shape, height, width)

    # BGR to RGB
    img = img[:, :, ::-1]

    return img, label

# Inner class of LSA.update_conv2d
class ScaledConv2d(nn.Conv2d): 
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        
        """ < weight_scaling >
        Initialize weight parameter which sized out_channel x 1 x 1 x 1 (to match dimension)
        Which means, there is one weight_scaling parameter per each filter
        """
        self.weight_scaling = nn.Parameter(torch.ones_like(torch.Tensor(out_channels, 1, 1, 1)))
        # self.reset_parameters()

    def reset_parameters(self):
        # The if condition is added so that the super call in init does not reset_parameters as well.
        if hasattr(self, 'weight_scaling'):
            nn.init.normal_(self.weight_scaling, 1, 1e-5) # 
            super().reset_parameters()
    
    # nn.Conv2d automatically returns result of forward() when called
    def forward(self, input):
        torch_version_str = str(torch.__version__).split('.')
        
        if int(torch_version_str[0]) >= 1 and int(torch_version_str[1]) > 7:
            # 2d convolution, but updated weight = weight_scaling * weight
            # In the case of convolution, 
            return self._conv_forward(input, self.weight_scaling * self.weight, self.bias) 
        else:
            ####################### CUSTOM ################################
            return self._conv_forward(input, self.weight_scaling * self.weight, self.bias) 
            # return self._conv_forward(input, self.weight_scaling * self.weight) ## reference, but wrong
            ###############################################################
    
    """ < _conv_forward in nn.Conv2d >
    
        def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
    """

# Inner class of LSA.update_linear
class ScaledLinear(nn.Linear):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)
        
        """ < weight_scaling >
        Initialize weight parameter which sized out_features x 1
        Which means, there is one weight_scaling parameter per each output node.
        i.e. if nn.Linear has 10 inputs and 5 outputs, then there are 5 weight_scaling parameters
        which mean, each 10 weights share single weight_scaling parameter.
        """
        self.weight_scaling = nn.Parameter(torch.ones_like(torch.Tensor(out_features, 1)))
        self.reset_parameters()

    def reset_parameters(self):
        # The if condition is added so that the super call in init does not reset_parameters as well.
            if hasattr(self, 'weight_scaling'):
                nn.init.normal_(self.weight_scaling, 1, 1e-5)
                super().reset_parameters()
                
    # nn.Linear automatically returns result of forward() when called
    def forward(self, input):
        
        # # FUCK THIS CODE # #
        # self.weight = nn.Parameter(self.weight)
        # self.bias = nn.Parameter(self.bias)
        # self.weight_scaling = nn.Parameter(self.weight_scaling)

        return F.linear(input, self.weight_scaling * self.weight, self.bias)

class LSA:
    def __init__(self,
                 original_model):

        self.mdl = copy.deepcopy(original_model)

    def update_conv2d(self, m, parent):
        lsa_update = ScaledConv2d(m[1].in_channels, m[1].out_channels, m[1].kernel_size, m[1].stride,
                                  m[1].padding, m[1].dilation, m[1].groups, None, m[1].padding_mode)
        lsa_update.weight, lsa_update.bias = m[1].weight, m[1].bias
        setattr(parent, m[0], lsa_update)

    def update_linear(self, m, parent):
        lsa_update = ScaledLinear(m[1].in_features, m[1].out_features)
        lsa_update.weight, lsa_update.bias = m[1].weight, m[1].bias
        setattr(parent, m[0], lsa_update)

    def add_lsa_params(self):
        
        # Original ##
        
        for m in self.mdl.named_children():
                # m[0] : name of each submodule / m[1] : Submodule itself
                if isinstance(m[1], nn.Conv2d) and m[1].weight.requires_grad:
                    self.update_conv2d(m, self.mdl)
                elif isinstance(m[1], nn.Linear) and m[1].weight.requires_grad:
                    self.update_linear(m, self.mdl)
                elif len(dict(m[1].named_children())) > 0: # Find if first submodule of m also contains submodule 
                    
                    for n in m[1].named_children(): # Find submodule of m[1]
                        if isinstance(n[1], nn.Conv2d) and n[1].weight.requires_grad:
                            self.update_conv2d(n, m[1])
                        elif isinstance(n[1], nn.Linear) and n[1].weight.requires_grad:
                            self.update_linear(n, m[1])
                        elif len(dict(n[1].named_children())) > 0:
                            
                            for o in n[1].named_children():
                                if isinstance(o[1], nn.Conv2d) and o[1].weight.requires_grad:
                                    self.update_conv2d(o, n[1])
                                elif isinstance(o[1], nn.Linear) and o[1].weight.requires_grad:
                                    self.update_linear(o, n[1])
                                elif len(dict(o[1].named_children())) > 0:
                                    
                                    for p in o[1].named_children():
                                        if isinstance(p[1], nn.Conv2d) and p[1].weight.requires_grad:
                                            self.update_conv2d(p, o[1])
                                        elif isinstance(p[1], nn.Linear) and p[1].weight.requires_grad:
                                            self.update_linear(p, o[1])
                                        elif len(dict(p[1].named_children())) > 0:
                                            
                                            for q in p[1].named_children():
                                                if isinstance(q[1], nn.Conv2d) and q[1].weight.requires_grad:
                                                    self.update_conv2d(q, p[1])
                                                elif isinstance(q[1], nn.Linear) and q[1].weight.requires_grad:
                                                    self.update_linear(q, p[1])
        return self.mdl
    
    '''
    adds LSA scaling parameters to conv and linear layers
        - max. nested object depth: 4
        - trainable_true (i.e. does not add LSA params to layers which are not trained, e.g. in classifier only training)
    '''
    
    """ < model.named_children() >
    
    --> Returns Direct Sub-Modules
    
    ex)
    
    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            return x

    model = MyModule()

    # Iterate over named children
    for m in MyModule.named_children():
        print(m)
        
        
    result)
    ('conv1', Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    ('relu1', ReLU())
    ('conv2', Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    ('relu2', ReLU())
    """
    
    """ < Description of Below Nested-Iteration >
    1. Find submodules using named_children (only direct-submodules are finded)
    2. Check if each submodule is instance of Conv2d or Linear, or recursively contains submodule
    3-1. If instance of Conv2d or Linear, call update_conv2d or update_linear
    3-2. If instance contains submodule, do recursively
    """
    
    """
    Customly modified to see what's going on
    You can change it to below version (it's original), 
    and also remove "print_moudule_info" function
    """
        
        # for m in self.mdl.named_children():
        #         # m[0] : name of each submodule / m[1] : Submodule itself
        #         if isinstance(m[1], nn.Conv2d) and m[1].weight.requires_grad:
        #             self.update_conv2d(m, self.mdl)
        #         elif isinstance(m[1], nn.Linear) and m[1].weight.requires_grad:
        #             self.update_linear(m, self.mdl)
        #             print(f"HELLO FROM DEPTH 0")
        #             print(m[1])
        #         elif len(dict(m[1].named_children())) > 0: # Find if first submodule of m also contains submodule 
                    
        #             for n in m[1].named_children(): # Find submodule of m[1]
        #                 if isinstance(n[1], nn.Conv2d) and n[1].weight.requires_grad:
        #                     self.update_conv2d(n, m[1])
        #                 elif isinstance(n[1], nn.Linear) and n[1].weight.requires_grad:
        #                     self.update_linear(n, m[1])
        #                     print(f"HELLO FROM DEPTH 1")
        #                 elif len(dict(n[1].named_children())) > 0:
                            
        #                     for o in n[1].named_children():
        #                         if isinstance(o[1], nn.Conv2d) and o[1].weight.requires_grad:
        #                             self.update_conv2d(o, n[1])
        #                         elif isinstance(o[1], nn.Linear) and o[1].weight.requires_grad:
        #                             self.update_linear(o, n[1])
        #                             print(f"HELLO FROM DEPTH 2")
        #                         elif len(dict(o[1].named_children())) > 0:
                                    
        #                             for p in o[1].named_children():
        #                                 if isinstance(p[1], nn.Conv2d) and p[1].weight.requires_grad:
        #                                     self.update_conv2d(p, o[1])
        #                                 elif isinstance(p[1], nn.Linear) and p[1].weight.requires_grad:
        #                                     self.update_linear(p, o[1])
        #                                     print(f"HELLO FROM DEPTH 3")
        #                                 elif len(dict(p[1].named_children())) > 0:
                                            
        #                                     for q in p[1].named_children():
        #                                         if isinstance(q[1], nn.Conv2d) and q[1].weight.requires_grad:
        #                                             self.update_conv2d(q, p[1])
        #                                         elif isinstance(q[1], nn.Linear) and q[1].weight.requires_grad:
        #                                             print(f"HELLO FROM DEPTH 4")
        #                                             self.update_linear(q, p[1])
        #     return self.mdl

    
def print_module_info(module, depth):
    indent = "  " * depth * 2
    print(f"{indent}This is Depth {depth}")
    print(f"{indent}Module Name: {module[0]}")
    print(f"{indent}Module Type: {module[1].__class__.__name__}\n")
    
    
""" 결론
1. All code of this file is about LSA
2. LSA is supported (at least implemented) to Linear, and Conv2d
3. There is no clue to find dataset-dependency (candidates : pytorch_model init)
"""