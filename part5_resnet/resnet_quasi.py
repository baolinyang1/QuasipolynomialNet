import torch
import torch.nn as nn

import torch.nn.functional as F  
from torchvision import models, transforms

# %%
class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(CustomConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

        self.exponent = nn.Parameter(torch.ones(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        pad = self.padding
        stride = self.stride
        kernel_size = self.kernel_size
        
        if pad > 0:
            x = F.pad(x, (pad, pad, pad, pad))

        out_height = (height + 2 * pad - kernel_size) // stride + 1
        out_width = (width + 2 * pad - kernel_size) // stride + 1

        x_unfold = F.unfold(x, kernel_size, stride=stride, padding=0)
        x_unfold = x_unfold.view(batch_size, in_channels, kernel_size, kernel_size, out_height, out_width)
        
        weight = self.weight.view(self.out_channels, self.in_channels, kernel_size, kernel_size, 1, 1)
        exponent = self.exponent.view(self.out_channels, self.in_channels, kernel_size, kernel_size, 1, 1)

        x_exp = x_unfold.unsqueeze(1) ** exponent

        out = (x_exp * weight).sum(dim=(2, 3, 4))
        
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)

        return out
    
class DefaultConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DefaultConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        pad = self.padding
        stride = self.stride
        kernel_size = self.kernel_size
        
        if pad > 0:
            x = F.pad(x, (pad, pad, pad, pad))

        out_height = (height + 2 * pad - kernel_size) // stride + 1
        out_width = (width + 2 * pad - kernel_size) // stride + 1

        x_unfold = F.unfold(x, kernel_size, stride=stride, padding=0)
        x_unfold = x_unfold.view(batch_size, in_channels, kernel_size, kernel_size, out_height, out_width)
        
        weight = self.weight.view(self.out_channels, self.in_channels, kernel_size, kernel_size, 1, 1)
        
        out = (x_unfold.unsqueeze(1) * weight).sum(dim=(2, 3, 4))
        
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)

        return out
    



class AddEpsilon(nn.Module):
    def __init__(self, epsilon=1e-10): # 1e-10 is smallest possible float in pytorch
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x + self.epsilon
#changes needed here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#use resnet architecture 
class Net(nn.Module):
    def __init__(self, convClass: nn.Module, random_seed: int):
        super().__init__()
        torch.manual_seed(random_seed)
        self.resnet = models.resnet18(pretrained=False)
        
        # Replace convolutional layers with custom layers
        self.replace_conv_layers(self.resnet, convClass)
        
        # Replace the fully connected layer for CIFAR-10
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)  # Assuming 10 classes for CIFAR-10

    def replace_conv_layers(self, model, convClass):
        # Function to replace all convolutional layers
        def replace_block(block, convClass):
            for name, module in block.named_children():
                if isinstance(module, nn.Conv2d):
                    # Replace Conv2d layer with convClass
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    kernel_size = module.kernel_size[0]
                    stride = module.stride[0]
                    padding = module.padding[0]
                    block.add_module(name, convClass(in_channels, out_channels, kernel_size, stride=stride, padding=padding))
                else:
                    replace_block(module, convClass)
        
        # Replace layers in the model
        replace_block(model, convClass)

    def forward(self, x):
        x = self.resnet(x)
        return x
    
    
#----------------------------------------

# class Net(nn.Module):
#     def __init__(self,convClass: nn.Module, random_seed: int):
#         super().__init__()
#         torch.manual_seed(random_seed)
#         self.conv2 = convClass(3, 8, 5)
#         torch.manual_seed(random_seed)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(8 * 14 * 14, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         return x 
    #----------------------------------------