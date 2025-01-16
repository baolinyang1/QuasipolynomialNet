# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
import torch.nn.functional as F  
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import traceback
# import torch
# import torch.nn as nn
# import torchvision
# from torchvision import transforms

# from torch.nn import functional as F

# import time
# import math
# import json

# import matplotlib.pyplot as plt

# from pathlib import Path

# import traceback
# import argparse


# %%

def create_data_loader(batch_size, device):
  # Split the provided CIFAR-10 train set (50,000 images) into your train and val sets
  # Use the first 40,000 images as your train set and the remaining 10,000 images as val set
  # Use all 10,000 images in the provided test set as your test set

  transform = transforms.Compose([
    transforms.ToTensor(),
      # This is the mean and the average of the cifar db because we are normalizing it.
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), 
  ])

  # load cifar
  train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

  # split train set into train and val
  train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000])

  # create data loaders
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=8)
  val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True,num_workers=8)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True,num_workers=8)

  train_loader = [(inputs.to(device), labels.to(device)) for inputs, labels in train_loader]
  val_loader = [(inputs.to(device), labels.to(device)) for inputs, labels in val_loader]
  test_loader = [(inputs.to(device), labels.to(device)) for inputs, labels in test_loader]


  return train_loader, val_loader, test_loader


# create folder data/models_and_boundaries/ if it does not exist
Path("./result_data/models_and_boundaries/").mkdir(parents=True, exist_ok=True)

# create folder data/experiments/ if it does not exist
Path("./result_data/experiments/").mkdir(parents=True, exist_ok=True)

print(torch.__version__)


from resnet_quasi import DefaultConv2d, CustomConv2d, Net



def initiate_experiment(model,device,batch_size=2):
    

    train_loader, val_loader, test_loader = create_data_loader(batch_size, device)

    criterion = nn.CrossEntropyLoss()


    return model, train_loader, val_loader, criterion


def train_model(model,train_loader, val_loader, criterion, optimizer, num_epochs,experiment_name,clamp_min, clamp_max, total_epochs=0):
    # start measuring time
    start = time.time()
    total_epochs = 0

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        # for each batch
        total_epochs += 1
        for i, (inputs, labels) in enumerate(train_loader):
            # count the number of weights with very small magnitute < 1e-3
            nans_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    nans_count += torch.sum(torch.isnan(param.grad)).item()
            
            # clamp exponents to avoid nans
            for name,param in model.named_parameters():
                if "exponent" in name:
                    param.data = torch.clamp(param.data, clamp_min, clamp_max)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print all gradients of the first synapse
            # print(f"synapse grads : {model.classifier[-1].synapses[0][0].main[0].grad}")
            # print(f"weights of first layer {model.classifier[-1].synapses[0][0]}")
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()


        print(f'Epoch {epoch+1}/{num_epochs}, total epochs is {total_epochs} Train Loss: {train_loss:.4f}, Train Accuracy: {train_correct/40000:.4f}')
        train_loss_list.append(train_loss)
        train_acc_list.append(train_correct/40000)

        # validate
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
        val_acc = val_correct/10000
        print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
    
    # plot the loss and accuracy curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label='val')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='train')
    plt.plot(val_acc_list, label='val')
    plt.title('Accuracy')
    plt.legend()

    plt.savefig(f"./result_data/models_and_boundaries/{experiment_name}.png")

    for name, param in model.named_parameters():
        if 'exponent' in name:
            print(f"{name}: {param.data}")

    
    # end measuring time
    end = time.time()
    time_taken = end - start
    print(f"Time taken: {time_taken:.2f}s")

    # save model to disk
    torch.save(model.state_dict(), f"./result_data/models_and_boundaries/{experiment_name}.pth")
        
    print(f"total epochs: {total_epochs}")

    # print all parameters of the model
    for name, param in model.named_parameters():
        print(f"{name} has shape {param.shape} and value {param}")

    # print model gradients
    for name, param in model.named_parameters(): 
        print(f"{name} has gradient {param.grad}")
        

    return total_epochs, time_taken, train_loss_list, train_acc_list, val_loss_list, val_acc_list

# %%

def main():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--random_seed", type=int, default=100, help="Set the random seed.")
    parser.add_argument("--clamp_min", type=int, default=1, help="Set the minimum clamp value.")
    parser.add_argument("--clamp_max", type=int, default=10, help="Set the maximum clamp value.")
    parser.add_argument("--cuda_gpu", type=int, default=None, help="Select the GPU on which to run.")
    parser.add_argument("--brief", type=str, default="", help="Select the GPU on which to run.")

    args = parser.parse_args()
    
    random_seed = args.random_seed
    clamp_min = args.clamp_min
    clamp_max = args.clamp_max
    cuda_gpu = args.cuda_gpu
    brief = args.brief
  
    device = "cpu"
    
    print("Random Seed:", random_seed)
    print("Clamp Min:", clamp_min)
    print("Clamp Max:", clamp_max)
    print("Brief", brief)
    if cuda_gpu is None:
        print("No CUDA GPU selected. Running on CPU.")
    elif not torch.cuda.is_available():
        print("CUDA GPU not available. Running on CPU.")
    else:
        print("CUDA GPU Selected:", cuda_gpu)
        device = torch.device(f"cuda:{cuda_gpu}")    

    print("Device:", device)

    all_models = [
      CustomConv2d
      ,DefaultConv2d
      ]  

    experiments = []

    clamp_range_str = f"{clamp_min}_{clamp_max}"
    EXPERIMENT_FILE = f"./result_data/experiments/{brief}.json"
    # check if the experiment file exists
    try:
        with open(EXPERIMENT_FILE, 'r') as file:
            experiments = json.load(file)
            print("loaded experiments from file")
    except FileNotFoundError:
        
        for model_idx,network in enumerate(all_models):
            experiment_name = f"{brief}_network_{network.__name__}"
            experiments.append({"name":experiment_name, "model":model_idx , "done":False , "seed": random_seed, "lr_different": False, "clamp_range": clamp_range_str})
        experiments.append({"name":f"{brief}_network_CustomConv2d_lr", "model":0 , "done":False , "seed": random_seed, "lr_different": True,"clamp_range": clamp_range_str })
        # print("created new experiments")

        
    for experiment in experiments:
            try:
                print(f"EXPERIMENT {experiment['name']} started")
                model = Net(all_models[experiment['model']], random_seed)
                model = model.to(device)
                if experiment['done']:
                    print(f"EXPERIMENT {experiment['name']} already done")
                    continue

                BATCH_SIZE = 2
                LEARNING_RATE = 0.005
                model, train_loder, val_loader, criterion = initiate_experiment(model, device, batch_size=BATCH_SIZE)
                

                print(f"batch size is {BATCH_SIZE} and learning rate is {LEARNING_RATE}")

                # divide parameters into groups: exponent, weight, bias
                bias_params = [param for name, param in model.named_parameters() if "bias" in name]
                weight_params = [param for name, param in model.named_parameters() if "weight" in name]
                exponent_params = [param for name, param in model.named_parameters() if "exponent" in name]
                optimizer = torch.optim.SGD([
                    {"params": bias_params, "lr": LEARNING_RATE},
                    {"params": weight_params, "lr": LEARNING_RATE},
                    {"params": exponent_params, "lr": LEARNING_RATE*10 if experiment['lr_different'] else LEARNING_RATE}
                ])

                
                total_epochs,time_taken, train_loss_list, train_acc_list, val_loss_list, val_acc_list  = train_model(model, train_loder,val_loader, criterion, optimizer, 100, experiment_name=experiment['name'],clamp_min=clamp_min, clamp_max=clamp_max)
                experiment['done'] = True
                experiment['total_epochs'] = total_epochs
                experiment['time_taken'] = time_taken
                experiment['max_val_accuracy'] = max(val_acc_list)
                experiment['max_train_accuracy'] = max(train_acc_list)
                print(f"EXPERIMENT {experiment['name']} finished")
                
            except:
                print("EXPERIMENT failed")
                print(traceback.format_exc())


    print(f"experiments status: {experiments}")

    # save experiments as a json file

    with open(EXPERIMENT_FILE, "w") as f:
        json.dump(experiments, f)
        
    print(experiments)

if __name__ == "__main__":
    main()
