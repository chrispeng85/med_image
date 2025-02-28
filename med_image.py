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
        
        

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.out_channels = out_channels

    def forward(self, x):
        residual  = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out) + self.skip(x)

        out = self.relu(out)

        return out


class Dropout(nn.Module):

    def __init__(self,p):
        super().__init__()
        self.p = p
        #dropout probability

    def forward(self, input):
        p = self.p
        bernoulli_distr = torch.distributions.bernoulli.Bernoulli(torch.tensor[p])
        index_zero = bernoulli_distr.sample(input.shape).squeeze()
        index_zero = index_zero.type(torch.BoolTensor)
        input[index_zero] = 0
        return 1/(1-p) * input

class BatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, input):

        eps = 1e-5
        batch_mean = torch.mean(input, 0)
        temp = torch.zeros_like(input)
        for i in range(input.shape[1]):
            temp[:, i, :] = (input[:, i, :] - batch_mean[i, :])**2
        batch_var = torch.mean(temp, 0)

        for i in range(input.shape[1]):
            input[:, i, :] = (input[:, i, :] - batch_mean[i, :])/torch.sqrt(batch_var[i, :] + eps)
            input[:, i, :] = self.gamma[i] * input[:, i, :] + self.beta[i]

        return input


class ResidualStack(nn.Module):
    def __init__(self, in_channels, out_channels, stride, num_blocks):
        super().__init__()
        self.residual_stack = nn.ModuleList([ResidualBlock(in_channels, out_channels, stride)])
        self.residual_stack.extend([ResidualBlock(out_channels, out_channels, 1) for i in range(1, num_blocks)])

    def forward(self, input):
        out = input
        for layer in self.residual_stack:
            out = layer(out)

        return out

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = fun
    def forward(self, x):
        return self.func(x)

class ResNet(nn.Module):
    def __init__(self, num_classes, n =2):
        super(ResNet, self).__init__()
        self.resnet = nn.Sequential(

            nn.Conv2d(3, 16 ,3, padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResidualStack(16,16,1,n),
            ResisualStack(16,32,2,n),
            ResidualStack(32,64,2,n),
            nn.AvgPool2d(8),
            Lambda(lambda x: x.squeeze)),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.resnet(x)


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
    
    data_dir = "/root/med_images/Knee Osteoarthritis Classification"
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






 


    

    







