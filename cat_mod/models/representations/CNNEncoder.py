import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.pool = nn.MaxPool2d(2, 2)  

        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  

        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  

    def forward(self, x):
        
        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        
        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        
        x = x.view(-1, 64 * 7 * 7)  

        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) 
        return x

    def encode(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        
        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        
        x = x.view(-1, 64 * 7 * 7)  

        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(
    model,
    criterion,
    optimizer,
    train_loader,
    num_epochs,
):
    train_loss_rec = []
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()

            
            outputs = model(images)

            
            loss = criterion(outputs, labels)

            
            loss.backward()

            
            optimizer.step()

            
            running_loss += loss.item()

        
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
        train_loss_rec.append(running_loss / len(train_loader))
        plt.plot(np.arange(epoch+1), train_loss_rec)



