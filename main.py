#Treinamento de uma rede neural CNN para classificação de estradas com base em imagens
#Comentar a função de treinamento depois de completo
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

#Adiciona mudificações às entradas, uniformizando-as e convertendo elas em vectores numéricos
transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

#---Criação dos dataloaders---
dataset = datasets.ImageFolder(root='dataset', transform=transforms)

train_size = int(0.8*len(dataset))

val_size = len(dataset) - train_size

train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

#Configurando a CNN
#Com duas camadas convolucionais, que são o suficiente para os nosso dados
class Vision(nn.Module):
    def __init__(self, num_classes=7):
        super(Vision, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.fc2(x)
    
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Vision().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#usado para salvar apenas o melhor modelo durante o treinamento
def train(num_epochs, best):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        #---treinamento---
        for input, target in train_loader:
            inputs, targets = input.to(device), target.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        
        total_loss = running_loss/len(train_loader)
        
        print(f"Época: {epoch+1}, loss: {total_loss}")

        #salva apenas se o erro diminuir
        if total_loss < best:
            best = total_loss
            torch.save(model.state_dict(), 'vision.pth')
            print("Modelo atualizado!")


        #Etapa de validação
        model.eval
        correct, total = 0,0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        print(f"Vall Acc: {100*(correct/total)}")
    print("treino completo!")

#train(30, float("inf"))