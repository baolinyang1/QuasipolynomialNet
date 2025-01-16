import torch
import numpy as np
from torch.autograd import Variable
from torch import FloatTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 2, 5, 1

x = Variable(FloatTensor(np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])), requires_grad=False)
y = Variable(FloatTensor(np.array([[0., 1., 1., 0.]])), requires_grad=False)

# Create two linear modules and initialize their weights
L1 = nn.Linear(D_in, H, bias=False)
L2 = nn.Linear(H, D_out, bias=False)
L1.weight.data.uniform_(0., 1.).type(dtype)
L2.weight.data.uniform_(0., 1.).type(dtype)

model = nn.Sequential(L1,
                      nn.Sigmoid(),
                      L2,
                      nn.Sigmoid())


print("W1: ", L1.weight.data)
print("W2: ", L2.weight.data)

optimizer = optim.SGD(model.parameters(), lr=1.)

success = False
for epoch in range(10000):

    layer2 = model(x)

    loss = F.mse_loss(layer2, y, size_average=False)

    worst_error = (y.t() - layer2).abs().max()
    if not success and worst_error.data.item() < .5:
        print("100% accuracy achieved in", epoch + 1, "epochs")
        success = True
    if worst_error.data.item() < .05:
        break

    if epoch % 100 == 0:
        print("Epoch %d: Loss %f  Predictions %s" % (
        epoch + 1, loss.data.item(), ' '.join(["%.3f" % p for p in (layer2.data.cpu().numpy())])))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print("Epoch %d: Loss %f  Predictions %s" % (
epoch + 1, loss.data.item(), ' '.join(["%.3f" % p for p in (layer2.data.cpu().numpy())])))

print("W1: ", L1.weight.data)
print("W2: ", L2.weight.data)