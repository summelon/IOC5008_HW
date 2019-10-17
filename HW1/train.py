import torch
import torch.nn as nn
import torchvision
import numpy as np
import time
import os
import copy
from PIL import Image

# Ignore the large image loading warning
Image.MAX_IMAGE_PIXELS = None


# Define a self dataset reader for dataset splitting
class SelfDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# Image data pre-processing
def dataset_pre_processing(data_dir='./data/train'):
    # Define training & validation transformer
    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.transforms.RandomResizedCrop(224),
            torchvision.transforms.transforms.RandomHorizontalFlip(),
            torchvision.transforms.transforms.ToTensor(),
            torchvision.transforms.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ]),
        'val': torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load image from folder
    whole_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir))

    # Split dataset into training dataset & validation dataset as 8:2
    train_split = .8
    split_number = int(train_split * len(whole_dataset))

    indices = list(range(len(whole_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    dataset_buffer = {'train': [], 'val': []}
    for i in range(split_number):
        dataset_buffer['train'].append(whole_dataset[indices[i]])

    for i in range(split_number, len(whole_dataset)):
        dataset_buffer['val'].append(whole_dataset[indices[i]])

    data_sets = {x: SelfDataset(dataset_buffer[x], transform=data_transforms[x])
                 for x in ['train', 'val']}

    data_set_size = {x: len(data_sets[x]) for x in ['train', 'val']}
    # Load data
    image_data_loader = {x: torch.utils.data.DataLoader(data_sets[x], batch_size=120,
                                                        num_workers=4, shuffle=True)
                         for x in ['train', 'val']}

    # Read class names
    class_names = whole_dataset.classes

    return data_set_size, image_data_loader, class_names


# Choose to fine tune parameters in hidden layer or not
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Read the pre-trained model
def initialize_model(num_classes, feature_extract, use_pre_trained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.

    model = torchvision.models.resnet34(pretrained=use_pre_trained)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


# Train & Validate
def train_model(model, criterion, optimizer, data_loader, data_set_size, num_epoch=25):
    # time counting
    since = time.time()

    # Initialize the best model weights.
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    loss_list = {'train': [], 'val': []}

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch+1, num_epoch))
        print('-' * 50)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data
            for inputs, labels in data_loader[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward and optimizer if only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Loss calculating
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Record loss & accuracy data
            epoch_loss = running_loss / data_set_size[phase]
            epoch_acc = running_corrects.double() / data_set_size[phase]
            loss_list[phase].append(epoch_loss)
            print('{:.11s} Loss:\t{:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print(data_set_size[phase])

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Best evaluation acc: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)

    return model, loss_list


# Fetch model parameters
def model_parameters(model, feature_extract=True):
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad is True:
                print("\t", name)

    return params_to_update


# ----------------------------------------------------------------------------------------------------------------------
# 30 epoch
epoch_number = 25
# Choose gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Read dataset and fetch the size, labels names and data loader
dataset_size, dataset_loader, classes_names = dataset_pre_processing()
# Load in the pre-trained model
model_ft = initialize_model(len(classes_names), feature_extract=True, use_pre_trained=True)
# Transform model to available device
model_ft = model_ft.to(device)
# Define optimizer, criterion
params = model_parameters(model_ft)
optimizer_ft = torch.optim.Adam(params)
criterion_ft = torch.nn.CrossEntropyLoss()

# Train & validate
trained_model, lossList = train_model(model_ft, criterion_ft, optimizer_ft,
                                      dataset_loader, dataset_size, epoch_number)

# Save the trained model
torch.save(trained_model.state_dict(), './model.pt')
