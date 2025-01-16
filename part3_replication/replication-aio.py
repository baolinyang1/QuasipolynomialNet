# %%
import torch
import torch.nn as nn

import traceback


import matplotlib.pyplot as plt

import time
import json

from torch.utils.data import TensorDataset, DataLoader

import common

from pathlib import Path

from common import Network1, Network2, Network3, Network4, Network5, Network6, Network24

# %%
import logging, sys
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.debug('A debug message!')

import getopt
dataset_idx = -1
try:
    dataset_idx = int(sys.argv[sys.argv.index("-d") + 1])
    seed = int(sys.argv[sys.argv.index("-s") + 1])
    print("Dataset index: ", dataset_idx)
    print("Seed: ", seed)
    torch.manual_seed(seed)
except ValueError:
    print("Invalid value for dataset index or seed. Both must be integers.")
    sys.exit(1)
except IndexError:
    print("Please provide both dataset index with flag -d and seed with flag -s")
    print("Usage: python script.py -d <dataset_index> -s <seed>")
    sys.exit(1)

# create folder data/models_and_boundaries/ if it does not exist
Path("data/models_and_boundaries/").mkdir(parents=True, exist_ok=True)

# create folder data/experiments/ if it does not exist
Path("data/experiments/").mkdir(parents=True, exist_ok=True)

# %%
torch.autograd.set_detect_anomaly(True)
device = "cpu"

# %%
# read the data from the "GeneratorInput.txt" file
def read_data(file_name):
    with open(file_name, 'r') as file:
        data = file.readlines()
    return data

raw_data = read_data("GeneratorInput.txt")

datasets = []
y_idx = 0
dataset = None
for line in raw_data:
    if len(line) <= 1:
        datasets.append(dataset.copy())
        y_idx = 0
        dataset = []
        continue
    y_idx += 1
    if dataset is None:
        dataset = []
    for x_idx,char in enumerate(line):
        if char == '0' or char == '1':
            dataset.append((y_idx,x_idx+1, int(char)))
    

# %%
def initiate_experiment(raw_dataset, model,batch_size=1):
    LEARNING_RATE = 1

    print(f"batch size is {batch_size} and learning rate is {LEARNING_RATE}")
    
    X = torch.tensor([list(sublist[:2]) for sublist in raw_dataset], dtype=torch.float32)
    y = torch.tensor([item for sublist in raw_dataset for item in sublist[2:]], dtype=torch.float32).unsqueeze(1)

    #scale the data to be between 0 and 1
    X = X / X.max()

    # Create a PyTorch dataset and data loader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    return model, X, y, dataloader, criterion, optimizer

def train_model(model, X, y ,dataloader, criterion, optimizer, num_epochs,experiment_name,total_epochs=0):
    # start measuring time
    
    start = time.time()
    total_epochs = 0

    # Train the model
    success = False
    for epoch in range(num_epochs):
        total_epochs += 1
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        # calculate accuracy for all training data
        y_pred = model(X)
        y_pred = torch.round(y_pred)
        correct = (y_pred == y).sum().item()
        accuracy = correct / len(y)

        if accuracy == 1:
            print(f"HOORAY, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}")
            success = True
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}")
            
    # end measuring time
    end = time.time()
    time_taken = end - start
    print(f"Time taken: {time_taken:.2f}s")

    if not success:
        experiment_name = f"{experiment_name}_failed"
    
    common.plot_decision_boundary(model, X, y, save_file_name=f"data/models_and_boundaries/{experiment_name}.png") 

    # save model to disk
    torch.save(model.state_dict(), f"data/models_and_boundaries/{experiment_name}.pth")
        
    print(f"total epochs: {total_epochs}")

    # print all parameters of the model
    for name, param in model.named_parameters():
        print(f"{name} has shape {param.shape} and value {param}")

    # print model gradients
    for name, param in model.named_parameters(): 
        print(f"{name} has gradient {param.grad}")
        

    return success, total_epochs, time_taken
        


# %%
all_models = [Network1, Network2, Network3, Network4, Network5, Network6, Network24]

experiments = []

EXPERIMENT_FILE = f"data/experiments/experiments_{dataset_idx+1}.json"
# check if the experiment file exists
try:
    with open(EXPERIMENT_FILE, 'r') as file:
        experiments = json.load(file)
        print("loaded experiments from file")
except FileNotFoundError:
    for model_idx,network in enumerate(all_models):
        experiment_name = f"dataset_{dataset_idx+1}_network_{network.__name__}"
        experiments.append({"name":experiment_name, "model":model_idx , "done":False , "dataset_idx": dataset_idx})
    print("created new experiments")
print(experiments)
        

# %%

for experiment in experiments:
        try:
            print(f"EXPERIMENT {experiment['name']} started")
            model = all_models[experiment['model']]()
            if experiment['done']:
                print(f"EXPERIMENT {experiment['name']} already done")
                continue
            model, X, y, dataloader, criterion, optimizer = initiate_experiment(datasets[experiment["dataset_idx"]], model)
            success, total_epochs,time_taken = train_model(model, X, y, dataloader, criterion, optimizer, 5000, experiment_name=experiment['name'])
            if success:
                experiment['done'] = True
                experiment['total_epochs'] = total_epochs
                experiment['time_taken'] = time_taken
                print(f"EXPERIMENT {experiment['name']} finished")
            else:
                print(f"EXPERIMENT {experiment['name']} max iterations reached")
            
        except:
            print("EXPERIMENT failed")
            print(traceback.format_exc())


print(f"experiments status: {experiments}")

# save experiments as a json file

with open(EXPERIMENT_FILE, "w") as f:
    json.dump(experiments, f)


