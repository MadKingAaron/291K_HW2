import torch
from torch import max_pool2d, nn, relu, optim
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import GoogleNet_Arz
import json

# checking if GPU is available or not
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class TestData(datasets.VisionDataset):

    filepath = './data/test.npy' #"/kaggle/input/ucsb-cs291k-hw2-2022-fall/test.npy"

    def __init__(self, root, transform=None,):
        super().__init__(root, transform)
        self.data = np.load(self.filepath)

    def __getitem__(self, index: int):

        img = self.data[index]

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)

class CNN_Layout1(nn.Module):
    def __init__(self) -> None:
        super(CNN_Layout1, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=1),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(16*5*5, 120), #Check input number
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
            #nn.Softmax()

        )

    def forward(self, x):
        y = self.model(x)
        return y

class CNN_Layout2(nn.Module):
    def __init__(self) -> None:
        super(CNN_Layout2, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1), # 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), # 3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=1), # 5
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1), # 6
            nn.Dropout(p=0.2),


            nn.Flatten(),
            nn.Linear(120*2*2, 120), #Check input number
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
            #nn.Softmax()

        )

    def forward(self, x):
        y = self.model(x)
        return y


def train(epoch, model, trainloader, criterion, optimizer):
    model.train()
    running_loss, total, correct = 0.0, 0, 0
    for i, data in tqdm(enumerate(trainloader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # compare predictions to ground truth
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # gather statistics
        running_loss += loss.item()

    running_loss /= len(trainloader)

    print('Training | Epoch: {}| Loss: {:.3f} | Accuracy on 50000 train images: {:.1f}'.format \
          (epoch+1, running_loss, 100 * correct / total))

def validate(epoch, model, valloader, criterion):
    model.eval()
    running_loss, total, correct = 0.0, 0, 0
    for i, data in tqdm(enumerate(valloader, 0)):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # compare predictions to ground truth
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # gather statistics
        running_loss += loss.item()

    running_loss /= len(valloader)

    print('Validation | Epoch: {}| Loss: {:.3f} | Accuracy on 10000 val images: {:.1f}'.format \
          (epoch+1, running_loss, 100 * correct / total))
    
    return {'loss':running_loss*100, 'accuracy':(100 * correct / total)}

def train_for_nepoch(nepoch, model, trainloader, criterion, optimizer):
    for epoch in range(nepoch):
        train(epoch, model, trainloader, criterion, optimizer)

def predict(model,testloader):

    model.eval()
    preds = []
    with torch.no_grad():
        # labels are not available for the actual test set
        for feature in tqdm(testloader):
            # calculate outputs by running images through the network
            outputs = model(feature.to(device))
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.tolist())

    return preds

def get_dataset(folder='./working_data'):
    train_ds = datasets.CIFAR10(folder, train=True, download=True)
    valid_ds = datasets.CIFAR10(folder, train=False)
    test_ds = TestData(folder)
    return {'train_ds':train_ds, 'valid_ds':valid_ds, 'test_ds':test_ds}

def transform_dataset(train_ds, valid_ds, test_ds):
    # first transform the images to tensor format, 
    # then normalize the pixel values
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_ds.transform = transform
    valid_ds.transform = transform
    test_ds.transform = transform

    return {'train_ds':train_ds, 'valid_ds':valid_ds, 'test_ds':test_ds}

def get_loaders(train_ds, valid_ds, test_ds):
    train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=64, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
    valid_ds, batch_size=1000
    )
    test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=1000
    )
    return {'train_loader':train_loader, 'valid_loader':valid_loader, 'test_loader':test_loader}

def test_training_and_val():
    model = CNN_Layout1().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 20

    ds = get_dataset()
    ds = transform_dataset(ds['train_ds'], ds['valid_ds'], ds['test_ds'])
    loaders = get_loaders(ds['train_ds'], ds['valid_ds'], ds['test_ds'])
    #print(loaders)
    train_for_nepoch(epochs, model, loaders['train_loader'], criterion, optimizer)
    validate(epochs-1, model, loaders['valid_loader'], criterion)


# Test different combinations


modelArch_list = [CNN_Layout1, CNN_Layout2]
optimizer_list = [optim.SGD, optim.Adam]
loss_list = [nn.CrossEntropyLoss]
#optimizer = [optim.SGD]
lrs = [0.001, 0.002, 0.003, 0.01, 0.02, 0.03]
epochs = [10, 20, 30, 40, 50, 60, 70]

    

def train_and_val_all_combo():
    ds = get_dataset()
    ds = transform_dataset(ds['train_ds'], ds['valid_ds'], ds['test_ds'])
    loaders = get_loaders(ds['train_ds'], ds['valid_ds'], ds['test_ds'])
    best_model_hyperpram = None
    best_model = None
    for modelArch in modelArch_list:
        for optimizerFunc in optimizer_list:
            for loss_func in loss_list:
                for epoch in epochs:
                    for lr in lrs:
                        print("Class:", modelArch.__name__, "Optimizer:", optimizerFunc.__name__, "LR:", lr, "Loss:", loss_func.__name__, "Epoch", epoch)
                        print("\nTraining:")
                        model = modelArch().to(device)
                        criterion = loss_func().to(device)
                        optimizer = optimizerFunc(model.parameters(), lr=lr)
                        train_for_nepoch(epoch, model, loaders['train_loader'], criterion, optimizer)
                        print("\nValidating:")
                        val_values = validate(epoch-1, model, loaders['valid_loader'], criterion)
                        
                        # Save best model performance
                        if best_model_hyperpram is None or best_model_hyperpram['accuracy'] < val_values['accuracy']:
                            best_model_hyperpram = {"class":modelArch.__name__, "optimizer":optimizerFunc.__name__, "lr":lr, "loss_func":loss_func.__name__, "epochs":epoch, "accuracy":val_values['accuracy'], 'total_loss':val_values['loss']}
                            best_model = model
                        print("\n\n\n")
    with open("best_model.json", "w") as outfile:
        json.dump(best_model_hyperpram, outfile, indent=4)
    torch.save(best_model, "best_model.pth")
    


#get_dataset()
#print(CNN_Layout1().to(device))

#test_training_and_val()
def test_trainingGoog():
    model = GoogleNet_Arz.GoogleNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 2

    ds = get_dataset()
    ds = transform_dataset(ds['train_ds'], ds['valid_ds'], ds['test_ds'])
    loaders = get_loaders(ds['train_ds'], ds['valid_ds'], ds['test_ds'])
    print(loaders)
    train_for_nepoch(epochs, model, loaders['train_loader'], criterion, optimizer)
    validate(epochs-1, model, loaders['valid_loader'], criterion)
def tester():
    ds = get_dataset()
    ds = transform_dataset(ds['train_ds'], ds['valid_ds'], ds['test_ds'])
    loaders = get_loaders(ds['train_ds'], ds['valid_ds'], ds['test_ds'])
    for data in loaders['train_loader']:
        print(data)
        print(data[0].shape)
        input()

#print(optim.SGD.__name__)
train_and_val_all_combo()
#test_trainingGoog()