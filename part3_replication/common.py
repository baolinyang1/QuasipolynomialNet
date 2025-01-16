import math
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

import logging, sys
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

class QuasiPolySynapse(nn.Module):
    def __init__(self, exponential_terms=3):
        super().__init__()
        power_initial_value = 1
        self.power = nn.Parameter(torch.tensor([power_initial_value], dtype=torch.float32), requires_grad=True)
        self.term_weights = nn.Parameter(torch.randn(exponential_terms), requires_grad=True)

    def forward(self, x):
        y = self.term_weights[0] * x ** self.power
        for i in range(1, min(int(torch.floor(self.power).item() - 1), len(self.term_weights))):
            y = y + self.term_weights[i] * x ** float(i)
        return y

    
    def __repr__(self):
        return f'QuasiPolySynapse(power={self.power}, term_weights={self.term_weights})'


class QuasiPolyLayer(nn.Module):
  def __init__(self, in_features, out_features, product=False, device="cpu"):
    super().__init__()
    self.out_features = out_features
    self.in_features = in_features
    self.bias_before = nn.Parameter(torch.randn(in_features), requires_grad=True)
    # a 2d module list of quasipoly synapses sizes in_features x out_features
    self.synapses = nn.ModuleList([nn.ModuleList([QuasiPolySynapse(3) for _ in range(self.in_features)]) for _ in range(self.out_features)])
    self.bias = nn.Parameter(torch.randn(out_features), requires_grad=True) # the radius
    self.product = product
    self.device = device
  
  def forward(self, x): # x is a batch of inputs


    y = torch.zeros(x.shape[0], self.out_features, dtype=torch.float32)
    y = y.to(self.device)
    
    # apply synapses to inputs
    for i in range(self.out_features):
        # print(f"synapse {i}")
        y[:, i] = 0

        for j in range(self.in_features):
            # logging.debug(f"y size is {y.shape} and x size is {x.shape}")
            logging.debug(f"before adding synapes={self.synapses[i][j]} y={y}")
            y[:, i] = y[:, i] + self.synapses[i][j](x[:, j])
                
            logging.debug(f"y in loop is {y} ")        

    logging.debug(f"y before bias is {y} ")
    # add bias
    y = y + self.bias

    logging.debug(f"y after bias is {y} ")

    return y
  
  def __repr__(self):
    return f'QuasiPolyLayer(in_features={self.in_features}, out_features={self.out_features}, product={self.product}, bias_before={self.bias_before}, synapses={self.synapses}, bias={self.bias})'

def plot_decision_boundary(model, X, y, save_file_name=""):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    grid = np.c_[xx.ravel(), yy.ravel()]

    with torch.no_grad():
        model.eval()
        output = model(torch.from_numpy(grid).float())
        preds = (output > 0.5).float()
        

    Z = preds.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y[:], cmap=plt.cm.Spectral)

    # save to file
    if len(save_file_name) > 0:
        plt.savefig(save_file_name, bbox_inches='tight')


class Network1(nn.Module):
    def __init__(self):
        super(Network1, self).__init__()
        self.fc1 = nn.Linear(2, 1)  # Linear synapse with 2 input features and 1 output feature
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x



class Network2(nn.Module):
    def __init__(self):
        super(Network2, self).__init__()
        self.fc1 = QuasiPolyLayer(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        logging.debug(f"out before sigmoid is {out}")
        out = self.sigmoid(out)
        logging.debug(f"out is {out}")
        return out
    





class Network3(nn.Module):
    def __init__(self):
        super(Network3, self).__init__()
        self.add_synapse = QuasiPolySynapse()
        self.product_synapse1 = QuasiPolySynapse()
        self.product_synapse2 = QuasiPolySynapse()
        self.bias1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.bias2 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s1 = self.add_synapse(x[:, 1])
        s1 = s1 + self.bias
        prod1 = self.product_synapse1(x[:, 0]) + self.bias1
        prod2 = self.product_synapse2(x[:, 1]) + self.bias2
        y = s1 + (prod1 * prod2) 
        y = self.sigmoid(y)
        return y.unsqueeze(-1)  # Ensure output has the same size as target
    
class Network31(nn.Module):
    def __init__(self):
        super(Network31, self).__init__()
        self.add_synapse = QuasiPolySynapse()
        self.product_synapse1 = QuasiPolySynapse()
        self.product_synapse2 = QuasiPolySynapse()
        self.bias1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.bias2 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s1 = self.add_synapse(x[:, 0])
        s1 = s1 + self.bias
        prod1 = self.product_synapse1(x[:, 0]) + self.bias1
        prod2 = self.product_synapse2(x[:, 1]) + self.bias2
        y = s1 + (prod1 * prod2) 
        y = self.sigmoid(y)
        return y.unsqueeze(-1)  # Ensure output has the same size as target

class Network4Infra(nn.Module):
    def __init__(self):
        super(Network4Infra, self).__init__()
        self.add_synapse1 = QuasiPolySynapse()
        self.add_synapse2 = QuasiPolySynapse()
        self.product_synapse1 = QuasiPolySynapse()
        self.product_synapse2 = QuasiPolySynapse()
        self.bias1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.bias2 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.bias3 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s1 = self.add_synapse1(x[:, 0])
        s2 = self.add_synapse2(x[:, 1])

        # Compute prod1
        prod1 = self.product_synapse1(x[:, 0])

        # Compute prod2
        prod2 = self.product_synapse2(x[:, 1])

        # Final product computation
        prod = s1 + s2 + prod1 * prod2 + self.bias3
        prod = prod.view(-1, 1)
        return prod
class Network4(nn.Module):
    def __init__(self):
        super(Network4, self).__init__()
        self.infra = Network4Infra()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        prod = self.infra(x)
        
        y = self.sigmoid(prod)

        return y.view(-1, 1)  # Ensure output has the same size as target
    
class Network24(nn.Module):
    def __init__(self):
        super(Network24, self).__init__()
        self.fc1 = QuasiPolyLayer(2, 2)
        
        self.module4 = Network4Infra()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.module4(out)
        out = self.sigmoid(out)
        return out


class Network41(nn.Module):
    def __init__(self):
        super(Network41, self).__init__()
        # first layer
        self.infra1_1 = Network4Infra()
        self.infra1_2 = Network4Infra()
        # second layer
        self.infra2_1 = Network4Infra()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        prod1 = self.infra1_1(x)
        prod2 = self.infra1_2(x)
        # concatenate the output of the two infrastructures
        y = torch.stack((prod1, prod2), 0)
        y = y.T
        # apply sigmoid activation
        y = self.sigmoid(y)
        y = self.infra2_1(y)
        y = self.sigmoid(y)

        return y.view(-1, 1)



class Network5(nn.Module):
    def __init__(self):
        super(Network5, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # First linear layer with 2 input features and 2 output features
        self.fc2 = nn.Linear(2, 1)  # Second linear layer with 2 input features and 1 output feature
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x



class Network6(nn.Module):
    def __init__(self):
        super(Network6, self).__init__()
        self.fc1 = QuasiPolyLayer(2, 2)  # First layer: 2 input features to 2 output features (hidden layer) with polynomial synapses
        self.fc2 = QuasiPolyLayer(2, 1)  # Second layer: 2 input features to 1 output feature (output layer) with polynomial synapses
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)  # Apply sigmoid activation to hidden layer output
        x = self.fc2(x)
        x = self.sigmoid(x)  # Apply sigmoid activation to final output
        return x

