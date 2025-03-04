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

os.environ["PYTORCH_CUDA_ALOC_CONF"] = "expandable_segments:True"

from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.cuda.amp import autocast, GradScaler


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





def train(ensemble_list, train_loader, test_loader, val_loader, criterion, num_epochs, device):

    total_step = len(train_loader)
    
   

    
    for epoch in range(num_epochs):

        for model, optimizer in ensemble_list:
            model.train()
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            

                del images, labels, outputs
            print('epoch [{}/{}], loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

            
        #individual eval
        for model, _ in ensemble_list: 
            model.eval()
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

        #ensemble eval
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                ensemble_outputs = torch.zeros(labels.size(0), outputs.size(1)).to(device)

                for model, _ in ensemble_list:
                    model.eval()
                    outputs = model(images)
                    ensemble_outputs += outputs

                _, predicted = torch.max(ensemble_outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'ensemble validation accuracy: {100 * correct / total:.2f}%')

        #eval on test 
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
    num_epochs = 64
    batch_size = 16
    learning_rate = 0.001

    import os
    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ensemble_list = []

    resnet = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = resnet.to(device)
    resnet_optim = torch.optim.SGD(resnet.parameters(), lr = learning_rate, momentum = 0.9)
    #scheduler = StepLR(resnet_optim, step_size = 7, gamma = 0.1)
    ensemble_list.append((resnet, resnet_optim))


    efficientnet = models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    efficientnet = efficientnet.to(device)
    efficient_optim = torch.optim.Adam(efficientnet.parameters(), lr= learning_rate)
    ensemble_list.append((efficientnet, efficient_optim))

    vit = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit = vit.to(device)
    vit_optim = torch.optim.SGD(vit.parameters(), lr = learning_rate, momentum = 0.9)
    ensemble_list.append((vit, vit_optim))


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
        transforms.Lambda(lambda x: x + torch.randn_like(x)*0.01),
        transforms.RandomErasing(p = 0.3, scale = (0.02, 0.1)),
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


    criterion = nn.CrossEntropyLoss()



    train(ensemble_list, train_loader, test_loader, val_loader, criterion, num_epochs, device)


if __name__ == "__main__":
    main()






 


    

    







