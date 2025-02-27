import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import os
from PIL import Image


from torch.utils.data.sampler import SubsetRandomSampler


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual  = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual 

        out = self.relu(out)

        return out
    

class Resnet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):

        super(Resnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3 , 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU()

        )
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1 )
        self.layer1 = self._make_layer(block ,128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x  

class dataset(Dataset):

    def __init__(self, x_data, y_data, transform = None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        img = self.x_data[idx]
        label = self.y_data[idx]

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2,0,1)
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).long()

        

        if self.transform:
            img = self.transform(img)

        return img, label
    

def train(model, train_loader, test_loader, val_loader, criterion, optimizer, num_epochs, device):

    total_step = len(train_loader)

    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
        print('epoch [{}/{}], loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('accuracy: {}%'.format(100 * correct / total))

    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print('accuracy on test images: {}%'.format( 100 * correct / total))




def main():

    num_classes = 3
    num_epochs = 20
    batch_size = 16
    learning_rate = 0.001

    import os
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Resnet(ResidualBlock, [3,4,6,3]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    
    data_dir = "/root/med_image/medical images/Knee Osteoarthritis Classification"
    categories = ['Normal', 'Osteopenia', 'Osteoporosis']
    splits = ['train', 'test', 'val']

    data = {split: {'x': [], 'y': []} for split in splits}

    for split in splits:

        for label, category in enumerate(categories):
            
            path = os.path.join(data_dir, split, category)


            image_files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]

            for img_file in image_files:
                img_path = os.path.join(path, img_file)
                img = Image.open(img_path)

                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img = img.resize((224,224))

                img_array = np.array(img)

                data[split]['x'].append(img_array)
                data[split]['y'].append(label)



            

        data[split]['x'] = np.array(data[split]['x'])
        data[split]['y'] = np.array(data[split]['y'])
    
    x_train, y_train = data['train']['x'], data['train']['y']
    x_test, y_test = data['test']['x'], data['test']['y']
    x_val, y_val = data['val']['x'], data['val']['y']

    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    y_val = torch.from_numpy(y_val).long()


    train_transforms = transforms.Compose([

        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear = 10, scale = (0.8,1.2)),
        transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
    ])

    val_transforms =  transforms.Compose([

        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])

    ])




    train_dataset = dataset(x_train, y_train,train_transforms)

    test_dataset = dataset(x_test, y_test, val_transforms)

    val_dataset = dataset(x_val, y_val, val_transforms)



    train_loader = DataLoader(

        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
        pin_memory = True

    )

    test_loader = DataLoader(

        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 4,
        pin_memory = True 
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 4,
        pin_memory = True
    )

    train(model, train_loader, test_loader, val_loader, criterion, optimizer, num_epochs, device)


if __name__ == "__main__":
    main()






 


    

    







