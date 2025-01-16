import re

string = "Weights of synapse (0, 0): Main: Parameter containing: tensor([-0.4151], requires_grad=True)x^Parameter containing: tensor([0.9729], requires_grad=True) Weights of synapse (0, 1): Main: Parameter containing: tensor([-0.6824], requires_grad=True)x^Parameter containing: tensor([1.0135], requires_grad=True) Weights of synapse (0, 2): Main: Parameter containing: tensor([-0.4089], requires_grad=True)x^Parameter containing: tensor([0.9964], requires_grad=True) Weights of synapse (0, 3): Main: Parameter containing: tensor([0.0800], requires_grad=True)x^Parameter containing: tensor([0.9973], requires_grad=True) Weights of synapse (0, 4): Main: Parameter containing: tensor([-0.3770], requires_grad=True)x^Parameter containing: tensor([1.0035], requires_grad=True)"

pattern = r"Parameter containing: tensor\(\[([-?\d.]+)\], requires_grad=True\)x\^Parameter containing: tensor\(\[([-?\d.]+)\], requires_grad=True\)"

matches = re.findall(pattern, string)

for base, power in matches:
    print(f"Base: {base}, Power: {power}")