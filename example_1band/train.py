from utils import CropedDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import torchvision 
import torch.nn.functional as F

path_base = r"/home/fm/Documents/GitHub/em_desenvolvimento/GIS_c/1dataset"

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0], [1]),
                                transforms.Resize((256, 256))])

face_dataset = CropedDataset(root_dir=path_base, transform=transform)

batch_size = 2
samples = face_dataset.__len__()

train_set, test_set = torch.utils.data.random_split(dataset=face_dataset, lengths = [samples-int(samples*0.4),int(samples*0.4) ])

train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#DATA

class CNN(nn.Module):
    def __init__(self,K):
        super(CNN,self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride = 2),
            nn.AvgPool2d(3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride = 2),
            nn.ReLU()
        )
        #self.weight_init()

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128*15*15,512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,K)
        )
    
    def forward(self,X):
        out = self.conv_layers(X)
        flatten = nn.Flatten()
        out = flatten(out)
        out = out.view(-1,out.shape[1])
        out = self.dense_layers(out)
        return out
    


K = face_dataset.n_classes +1
model = CNN(K)
model.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
model.double()
epochs=10

def batch_gd(model, criterion, optimizer, train_loader, test_loader,epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in tqdm(range(epochs)):
        train_loss = []
        
        for inputs,targets in train_loader:

            #Dados para GPU
            inputs,targets = inputs.to(device),targets.to(device)
            
            #Zerar valores de gradiente local
            optimizer.zero_grad()

            outputs = model(inputs)
            
            loss = criterion(outputs, targets)

            #Backward prop
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        train_loss=np.mean(train_loss)

        test_loss = []
        for inputs, targets in test_loader:
            inputs,targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        #Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss
    return train_losses,test_losses

train_losses,test_losses = batch_gd(model, criterion, optimizer, train_loader, test_loader,epochs)
plt.plot(train_losses, label = 'losses treino')
plt.plot(test_losses, label = 'losses teste')
plt.legend()
plt.show()

#Test accuracy
n_correct = 0
n_total = 0

for inputs,targets in train_loader:
    inputs,targets = inputs.to(device),targets.to(device)

    outputs = model(inputs)

    _,predictions = torch.max(outputs,1)

    #update max
    n_correct += (predictions ==targets).sum().item()
    n_total += targets.shape[0]

train_acc = n_correct / n_total

#test_acc
n_correct = 0
n_total = 0

for inputs,targets in test_loader:
    inputs,targets = inputs.to(device),targets.to(device)

    outputs = model(inputs)

    _,predictions = torch.max(outputs,1)

    #update max
    n_correct += (predictions ==targets).sum().item()
    n_total += targets.shape[0]

test_acc = n_correct / n_total
