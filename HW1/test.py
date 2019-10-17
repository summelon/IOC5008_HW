import torch
import torch.nn as nn
import torchvision
import csv
import os

# Define classes
classes_names = ('bedroom',
                 'coast',
                 'forest',
                 'highway',
                 'insidecity',
                 'kitchen',
                 'livingroom',
                 'mountain',
                 'office',
                 'opencountry',
                 'street',
                 'suburb',
                 'tallbuildin')


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


# Image data pre-processing
def dataset_pre_processing(data_dir='./data/'):
    # Define training & validation transformer
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Read image #name from test folder
    name = []
    file_list = os.listdir(data_dir+'test')
    for imgFile in file_list:
        name.append(os.path.splitext(imgFile)[0])
    name.sort()

    # Load data
    image_dataset = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir), data_transform)
                     for x in ['test', 'train']}

    image_data_loader = torch.utils.data.DataLoader(image_dataset['test'], batch_size=50,
                                                    shuffle=False, num_workers=0)

    print('Image Loaded.')

    return name, image_data_loader


# Image prediction
def data_predict(model, data_loader):
    # Prediction list
    result = []

    # Load image data
    for inputs, labels in data_loader:
        inputs = inputs.to(device)

        # Forward
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu()
            for p in range(len(preds)):
                index = preds[p].numpy()
                result.append(classes_names[index])

    print('Prediction Loaded')

    return result


# Write names and results into csv
def csv_writer(name, result):
    header = ['id', 'label']
    with open('./prediction.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(zip(name, result))

    print('Write in over.')


# Choose gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Read dataset and fetch the size, labels names and data loader
image_names, image_loader = dataset_pre_processing()
# Load in the pre-trained model and transform to available device
model_ft = initialize_model(len(classes_names), feature_extract=True).to(device)
# Load trained parameters from 'model.pt'
model_ft.load_state_dict(torch.load('./model.pt', map_location=device))
# Test mode
model_ft.eval()
# Record predictions and write to csv file
prediction_results = data_predict(model_ft, image_loader)
csv_writer(image_names, prediction_results)
