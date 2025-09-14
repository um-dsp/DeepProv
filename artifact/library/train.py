#This will create/configure/train and test the model then save it in "mnist_model.h5" in the same directory as the code running.
#MLP Multilayer Perceptron neural network - MNIST - A neuron in layer N is connected to all neurons in layer N+1
# Imports
from keras.datasets import mnist,cifar10
from keras.models import Sequential
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import Linear,GATConv,HGTConv
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout,MaxPooling2D
from library.utils import get_dataset, generate_attack,gen_train_batch #This is needed because mnist has 10 classes - classification problem
from keras.models import load_model
from keras.layers import BatchNormalization
import torch.nn as nn
import torch.nn.functional as F
import ember
import torch.optim as optim
from torchvision import datasets, transforms
import keras
import pickle
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import torch
import torch.optim as optim
from tqdm import tqdm
import copy
import pickle
from torch_geometric.data import Data, DataLoader
import dgl
from torch_geometric.utils import from_networkx
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GraphConv
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    gpu_num=0
# We definie the GNN model 
# Define your MessagePassing layer
class MessagePassing_Conv(MessagePassing):
    def __init__(self, in_channels, out_channels,**kwargs):
        super(MessagePassing_Conv, self).__init__(aggr="mean")
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return self.linear(aggr_out)
# Define your GNN model
class GNN_Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes,**kwargs):
        super(GNN_Classifier, self).__init__()
        self.conv1 = MessagePassing_Conv(in_dim, hidden_dim)
        self.conv2 = MessagePassing_Conv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, x,edge_index,edge_weight=None):
       
        # Apply message passing layers and activation.
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Calculate graph representation by global mean pooling
        x = torch_geometric.utils.scatter(x, batch,reduce='mean')
        x = torch.relu(x)

        # Classification layer
        x = self.classify(x)

        return x



def train_Cifar10(model_name):
    
    '''Cifar10 classification model using keras'''
    
    feature_vector_length = 32*32*3
    num_classes = 10

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    X_train = X_train.reshape((X_train.shape[0],32,32,3))
    X_test = X_test.reshape((X_test.shape[0],32,32,3))

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    Y_train = to_categorical(Y_train, num_classes)
    Y_test = to_categorical(Y_test, num_classes)
    
    input_shape = (feature_vector_length,)
 
    # Create the model
    model = Sequential()

    model.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(128,(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(Conv2D(128,(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(10 , activation = 'softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=10, batch_size=250, verbose=1, validation_split=0.2) # 20% - 80% training and 20%optimization (training samples)


    evaluate(model,X_test,Y_test)
    model.save("./models/"+model_name+"_2.h5")
    print("Trained and Saved CIFAR10 Model")
    
def evaluate(model,X_test,Y_test):
    '''evaluate a keras model'''
    try:
        test_results = model.evaluate(X_test, Y_test, verbose=1)
    except:
        Y_test=tf.math.argmax(Y_test,axis=1)
        test_results = model.evaluate(X_test, Y_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
 
   

def train_ember():
    
    '''train Ember classification model'''
    
    X_train, y_train, X_test, y_test = ember.read_vectorized_features("./data/ember2018/")
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(f'X_train :{X_train.shape}')
    print(f'y_train : {y_train.shape}')
    print(f'X_test  : {X_test.shape}')
    print(f'y_test  :{y_test.shape}')


    batch_size = 500
    epochs = 5
    model = Sequential()

    model.add(Dense(4608, activation='relu', input_shape=(2381,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(3584, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(3072, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(2560, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(1536, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(2, activation='sigmoid'))
    model.summary()
    model.compile(keras.optimizers.Adam(lr=1e-4),loss='binary_crossentropy', metrics=['accuracy'])# cross-entropy
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
            verbose=1, validation_data=(X_test, y_test))
    model.save("./models/Ember_1_2.h5")
    
    #validator = Validator(1,1)
    #validator.accuracy_from_file('./adversarial/cuckoo/CKO/cuckoo_1')
def get_model(name,model_type="keras"):
    if(name not in ['cifar10_2','cuckoo_1','ember_1','mnist_1','mnist_2','mnist_3']):
         raise ValueError('Model Not Supported')
    if model_type=="keras":
        model=load_model("./models/"+name+".h5")
    else:
        if "cifar10" in name:
            model=resnet("resnet18", num_classes=10)
        elif "mnist" in name:
            if "1" in name:
                model=NeuralMnist()
            elif "2" in name:
                model=MNIST_CNN()
        elif "cuckoo" in name:
            model=MLPClassifierPyTorch()
        elif "ember" in name:
            model = Ember_model()
            # load
            ckpt = torch.load("models/ember_best.pt")
            state = ckpt.get("state_dict", ckpt)
            if next(iter(state)).startswith("module."):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model.load_state_dict(state)
            return model
        model.load_state_dict(torch.load("./models/"+name+".pth"))
    return model

class NeuralMnist(nn.Module):

    '''PyTorch Implementation of Mnist model'''

    def __init__(self):
        super(NeuralMnist, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 350)
        self.fc2 = nn.Linear(350, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    def get_activation_functions(self):
        activation_functions = {}
        activation_functions['0'] = 'None'
        activation_functions['fc1'] = nn.ReLU()
        activation_functions['fc2'] = nn.ReLU()
        activation_functions['fc3'] = 'None'  # Assuming no activation function

        return activation_functions

class Ember_model(nn.Module):
    def __init__(self):
        super(Ember_model, self).__init__()
        self.input_layer = nn.Linear(2381, 128)
        self.hidden_layer1 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)  # Batch Normalization layer
        self.dropout1 = nn.Dropout(0.1)
        self.hidden_layer2 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)  # Batch Normalization layer
        self.dropout2 = nn.Dropout(0.1)
        self.hidden_layer3 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)  # Batch Normalization layer
        self.dropout3 = nn.Dropout(0.1)
        self.output_layer = nn.Linear(16, 1)


    def forward(self, x):
        x = F.relu((self.input_layer(x)))
        x=self.hidden_layer1(x)
        x = F.relu(self.bn2(x))
        x = self.dropout1(x)
        x=self.hidden_layer2(x)
        x = F.relu(self.bn3(x))
        x = self.dropout2(x)
        x=self.hidden_layer3(x)
        x = F.relu(self.bn4(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.output_layer(x))
        return x
    def get_activation_functions(self):
        activation_functions = {}
        activation_functions['input_layer'] =  nn.ReLU()
        activation_functions['hidden_layer1'] = 'None'
        activation_functions['bn2'] = nn.ReLU() # Assuming no activation function
        activation_functions['pool1'] = 'None'
        activation_functions['dropout1'] = 'None'
        activation_functions['hidden_layer2'] = 'None'
        activation_functions['bn3'] = nn.ReLU()
        activation_functions['dropout2'] = 'None'
        activation_functions['hidden_layer3'] = 'None'
        activation_functions['bn4'] = nn.ReLU()
        activation_functions['dropout3'] = 'None'
        activation_functions['output_layer'] = 'None'  # Assuming no activation function

        return activation_functions


# Define the model
class MLPClassifierPyTorch(nn.Module):
    def __init__(self):
        super(MLPClassifierPyTorch, self).__init__()
        self.fc1 = nn.Linear(in_features=1549, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=18)
        self.fc3 = nn.Linear(in_features=18, out_features=12)
        self.fc4 = nn.Linear(in_features=12, out_features=1)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation in the last layer for binary classification
        return F.sigmoid(x)
    
    def get_activation_functions(self):
        activation_functions = {}
        activation_functions['fc1'] = nn.ReLU()
        activation_functions['fc2'] = nn.ReLU()
        activation_functions['fc3'] = nn.ReLU()
        activation_functions['fc4'] = 'None'  # Assuming no activation function

        return activation_functions
class Deep(nn.Module):
    def __init__(self,nb_nodes):
        super().__init__()
        self.layer1 = nn.Linear(nb_nodes, 256)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Sequential( 
            nn.Dropout(), 
            nn.Linear(256, 128))
        self.act2 = nn.ReLU()
        self.layer3 = nn.Sequential( 
            nn.Dropout(),
            nn.Linear(128, 64))
        self.act3 = nn.ReLU()
        self.output = nn.Sequential( 
            nn.Dropout(),
            nn.Linear(64, 1))
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
class Cifar10_Net(nn.Module):

    '''Cifar10 pytorch model'''

    def __init__(self):
        super(Cifar10_Net, self).__init__()
        # First Conv Block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.4)

        # Second Conv Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.4)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1152, 1024)  # Adjust the input features size accordingly
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        # Conv Block 1
        x=self.conv1(x)
        x=F.relu(self.bn1(x))
        x=self.conv2(x)
        x=F.relu(self.bn2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Conv Block 2
        x=self.conv3(x)
        x = F.relu(self.bn3(x))
        x=self.conv4(x)
        x=F.relu(self.bn4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten and Fully Connected Layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    def get_activation_functions(self):
        activation_functions = {}
        activation_functions['conv1'] = 'None'
        activation_functions['bn1'] = nn.ReLU()
        activation_functions['conv2'] = 'None'
        activation_functions['bn2'] = nn.ReLU() # Assuming no activation function
        activation_functions['pool1'] = 'None'
        activation_functions['dropout1'] = 'None'
        activation_functions['conv3'] = 'None'
        activation_functions['bn3'] = nn.ReLU()
        activation_functions['conv4'] = 'None'
        activation_functions['bn4'] = nn.ReLU()
        activation_functions['pool2'] = 'None'
        activation_functions['dropout2'] = 'None'
        activation_functions['flatten'] = 'None'
        activation_functions['fc1'] = nn.ReLU()
        activation_functions['fc2'] = nn.ReLU()
        activation_functions['fc3'] = 'None'  # Assuming no activation function

        return activation_functions
def train_mnist_pytorch(path=None):

    # Create a training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the model
    model = NeuralMnist()
    model.to(device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Training code
    train_loader, test_loader = get_dataset('mnist',model_type='pytorch')
    print('Training code')
    for epoch in range(10):
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
    
    # evaluate model
    evaluate_model(model, test_loader,device=device)

    # save the model
    if path is not None:
        torch.save(model.state_dict(), path)
import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalization(nn.Module):
    """
    Standardizes the input data.
    Arguments:
        mean (list): mean.
        std (float): standard deviation.
        device (str or torch.device): device to be used.
    Returns:
        (input - mean) / std
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        num_channels = len(mean)
        self.mean = torch.FloatTensor(mean).view(1, num_channels, 1, 1)
        self.sigma = torch.FloatTensor(std).view(1, num_channels, 1, 1)
        self.mean_cuda, self.sigma_cuda = None, None

    def forward(self, x):
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.sigma_cuda = self.sigma.cuda()
            out = (x - self.mean_cuda) / self.sigma_cuda
        else:
            out = (x - self.mean) / self.sigma
        return out


class BasicBlock(nn.Module):
    """
    Implements a basic block module for Resnets.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Implements a basic block module with bottleneck for Resnets.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet model
    Arguments:
        block (BasicBlock or Bottleneck): type of basic block to be used.
        num_blocks (list): number of blocks in each sub-module.
        num_classes (int): number of output classes.
        device (torch.device or str): device to work on. 
    """
    def __init__(self, block, num_blocks, num_classes=10, device='cpu'):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out=self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    def get_activation_functions(self):
        activation_functions = {}
        activation_functions['conv1'] = 'None'
        activation_functions['bn1'] = nn.ReLU()
        activation_functions['layer1'] = 'None'
        activation_functions['layer2'] = 'None'
        activation_functions['layer3'] = 'None'
        activation_functions['layer4'] = 'None'
        activation_functions['layer5'] = 'None'
        return activation_functions
def resnet(name, num_classes=10, pretrained=False, device='cpu'):
    """
    Returns suitable Resnet model from its name.
    Arguments:
        name (str): name of resnet architecture.
        num_classes (int): number of target classes.
        pretrained (bool): whether to use a pretrained model.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    if name == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, device=device)
    elif name == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, device=device)
    elif name == 'resnet50':
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, device=device)
    elif name == 'resnet101':
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, device=device)
    
    raise ValueError('Only resnet18, resnet34, resnet50 and resnet101 are supported!')
    return

def evaluate_model(model, test_loader,device='cpu'):

    '''Evaluate a PyTorch model'''
    model.eval()  # Set the model to evaluation mode
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for data in tqdm(test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total += labels.size(0)
            if images.shape[-1]==2381:
                predicted=((outputs) > 0.5).float()
                predicted=predicted.view(-1).cpu().numpy()
                labels=labels.view(-1).cpu().numpy()
            elif outputs.data.shape[1]==1:
                predicted=((outputs) > 0.5).float()
            else:
                _, predicted = torch.max(outputs.data, 1)
            
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test inputs: {accuracy}%')
def train_cuckoo(train_loader):
    model = MLPClassifierPyTorch()
    criterion = nn.BCEWithLogitsLoss()  # Updated loss function for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    NUM_EPOCHS = 5
    for epoch in range(NUM_EPOCHS):
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')
class BinaryClassification(nn.Module):
    def __init__(self,nb_nodes):
        super(BinaryClassification, self).__init__()
        
        self.layer_1 = nn.Linear(nb_nodes, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.layer_out(x))
        
        return x
    
def compute_mismatch(model,X_test,Y_test):
    ones= 0
    zeros = 0
    correct = 0 
    pred = model.predict(X_test, verbose=0)
    for x,y in zip(pred,Y_test):
        if(np.argmax(x) == np.argmax(y)):
            correct+=1
        if(np.argmax(x) != np.argmax(y)):
            if(np.argmax(x)== 1):
                ones+=1
            if(np.argmax(x)== 0):
                zeros+=1
    print(f' mismatches 0 {zeros}' )        
    print(f' mismatches 1 {ones}' )        
    print(f' Accuracy  {correct/len(X_test) * 100}' )    

'''
def train_Detection_Model (X,Y):
    Y = to_categorical(Y)
    model = NeuralNetCifar_10()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    accuracy = 0
    #while(accuracy <80):
    X_train, X_test,y_train, y_test = train_test_split(X,Y ,random_state=104, test_size=0.25, shuffle=True)
    print(f'X_test len {len(X_test)}')
    print(f'Y_test len {len(y_test)}')

    accuracy = 1
        #Train Model
    while(accuracy < 95):
        #Load in the data in batches using the train_loader object
        correct =0
        for x, y in  zip(X_train,y_train):  

            y= torch.tensor(y)
            x = x[None, :]
            # Forward pass
            outputs = model(x)
            outputs = torch.squeeze(outputs)

            #print(torch.argmax(outputs),torch.argmax(y))

            loss = criterion(outputs, y)
            correct += 1 if (torch.argmax(outputs) == torch.argmax(y)) else 0 
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'accuracy {correct/len(X_train) *100}')
        accuracy = correct/len(X_train) *100

    torch.save(model, './advDetectionModels/torch_Cifar10_2.pt')
'''

def binary_acc(y_pred, y_test):
    #y_pred_tag = torch.round(torch.sigmoid(y_pred))
    y_pred_tag = torch.round(y_pred)
    
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

import torch
from torch import nn
from torch_geometric.nn import HGTConv, Linear, global_mean_pool

class HGT(nn.Module):
    def __init__(self, hidden_channels, metadata, out_channels, num_heads=2, num_layers=2):
        super().__init__()
        node_types, _ = metadata

        # One input projection per node type; -1 = lazy infer in_channels.
        self.lin_dict = nn.ModuleDict({
            nt: Linear(-1, hidden_channels) for nt in node_types
        })

        # Stack HGTConv layers (do NOT pass aggr here)
        self.convs = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, heads=num_heads)
            for _ in range(num_layers)
        ])

        self.lin_out = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        # 1) Project each node type to hidden dim
        x_dict = {nt: self.lin_dict[nt](x).relu() for nt, x in x_dict.items()}

        # 2) HGT layers; keep previous features when a type gets no messages
        for conv in self.convs:
            new_x = conv(x_dict, edge_index_dict)          # may have None for some types
            x_dict = {nt: (new_x.get(nt, None) if new_x.get(nt, None) is not None else x)
                      for nt, x in x_dict.items()}

        # 3) Graph-level readout: pool per node type, then sum (or concat)
        pooled = []
        for nt, x in x_dict.items():
            if x is None:
                continue
            pooled.append(global_mean_pool(x, batch_dict[nt]))

        # combine node-type representations (sum is common for HGT)
        hg = torch.stack(pooled, dim=0).sum(dim=0) if len(pooled) > 1 else pooled[0]

        # 4) Final classifier/regressor head
        return self.lin_out(hg)
def train_on_activations(X_train,Y_train,X_test,Y_test,model_name,model_path):
    
    if os.path.exists(model_path):
        print('loading existing model ...')
        model =torch.load(model_path)
    else:
        model=BinaryClassification(X_train.shape[1])
        '''
        if model_name=='cifar10_1':
            model = NeuralNetCifar_10()
        elif model_name=='Ember_2 ':
            model = NeuralNetCifar_10()
        elif model_name=='mnist_1':
            #model=NeuralNetMnist_1()
            model=Deep(X.shape[1])
        elif model_name=='mnist_2':
            model = NeuralNetMnist_2
        elif model_name=='mnist_3':
            model = NeuralNetMnist_3
        elif model_name=='cuckoo_1':
            model = NeuralNetCuckoo_1
        '''

    #Y = to_categorical(Y)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss() 
    #criterion = nn.BCEWithLogitsLoss()

    # shuffling training data
    X_train, _, y_train, _ = train_test_split(X_train,Y_train ,random_state=104, test_size=1, shuffle=True)
    # Standersize data
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
    
    if torch.cuda.is_available():
        model=model.cuda()
        
    X_test=torch.Tensor(X_test)
    Y_test=torch.Tensor(Y_test)  
    #print('[Graph MODEL TRAINING]')
    #print(f'X_train len {X_train.shape}')
    #print(f'Y_train len {y_train.shape}')
    #print(f'X_test len {X_test.shape}')
    #print(f'Y_test len {Y_test.shape}')

    n_epoch = 30 # number of epochs to run
    batch_size = 100#00  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)
    epoch=1
    
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    while(epoch < n_epoch):
        model.train()
        
        with tqdm.tqdm(batch_start, unit="batch") as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                
                
                # take a batch
                x = X_train[start:start+batch_size]
                y = y_train[start:start+batch_size]
                if x.shape[0]==1:
                    continue
                x=torch.Tensor(x)
                y=torch.Tensor(y)
                
                if torch.cuda.is_available():
                    x=x.cuda()
                    y=y.cuda()
                
                optimizer.zero_grad()
                
                #print(type(x))
                y_pred = model(x)
                
                loss = criterion(y_pred, y.unsqueeze(1))
                #acc = binary_acc(y_pred, y.unsqueeze(1))
                
                loss.backward()
                optimizer.step()
                
                '''
                # Forward pass
                outputs = model(x)[:,0]#.detach()[0]
                loss = criterion(outputs, y)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print progress
                acc = (outputs.round() == y).float().mean()
                '''
                bar.set_postfix(
                    loss=float(loss),
                    #train_acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        if torch.cuda.is_available():
            X_test=X_test.cuda()
            Y_test = Y_test.cuda()
        y_pred = model(X_test)
        acc = binary_acc(y_pred, Y_test.unsqueeze(1))#(y_pred.round() == Y_test).float().mean()
        acc = float(acc)
        print('Current test accuracy: ',float(acc))
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
        epoch+=1
        
    # restore and save model
    model.load_state_dict(best_weights)
    torch.save(model, model_path)   
    print(f' Best Accuracy : {best_acc}% Saved to {model_path}')

    return model
# Create a DataLoader for batch processing

def model_output(model,graphs_batch,data_type):
    global batch
    if data_type=="hetero":
        batch=[graphs_batch[i].batch for i in graphs_batch.node_types]
        out = model(
            graphs_batch.x_dict,
            graphs_batch.edge_index_dict,
            {nt: graphs_batch[nt].batch for nt in graphs_batch.node_types}
        )

    else:
        batch=graphs_batch.batch
        out = model(graphs_batch.x,graphs_batch.edge_index)
    return(out)

    

#train_GNN Train a GNN and save it 
def train_GNN(model,opt,num_epochs,batch_size,data_size,model_name,model_type,attack,dataset,folder,save=False):
    loss_his=[]
    acc_his=[]
    if "HGT" in str(type(model)):
        data_type="hetero"
    else:
        data_type="homo"
    # Training loop
    if torch.cuda.is_available():

        model=model.cuda()

    for epoch in tqdm(range(num_epochs)):
        pred_acc = 0
        loss_va = 0
        counter = 0
        for nb_bt in range(data_size):
            dataloader=gen_train_batch(dataset,model_name,attack,nb_bt,folder)
            dataloader = DataLoader(dataloader, batch_size=batch_size, shuffle=True)

            for batch_graphs ,batch_labels in dataloader:
                if torch.cuda.is_available():
                    batch_graphs=batch_graphs.cuda()
                # Convert batch of DGL graphs to PyTorch Geometric Data objects
                # Prepare features and labels
                labels = batch_labels
                logits = model_output(model,batch_graphs,data_type)
                logits=logits.cpu()
                loss = F.cross_entropy(logits, labels)

                # Calculate accuracy
                predicted_labels = torch.argmax(logits, dim=1)  # Get the class with the highest probability
                true_labels = torch.argmax(labels, dim=1)      # Get the true class labels

                correct_predictions = (predicted_labels == true_labels).sum().item()
                pred_acc += correct_predictions
                acc = correct_predictions / len(labels)

                loss_his.append(loss.item())
                acc_his.append(acc)

                # Backpropagation and optimization
                opt.zero_grad()
                loss.backward()
                opt.step()

                loss_va += loss.item() * len(labels)
                counter += len(labels)

        loss_va = loss_va / counter
        accuracy = pred_acc / counter

        # Print the loss and accuracy for each epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss_va:.2f}, Accuracy: {accuracy * 100:.2f}%')
    model_path=""
    if save:
        model_path="models/GNN_"+model_name+"_"+attack+"_"+model_type
        torch.save(model.state_dict(), model_path)
    return(model,acc_his,loss_his,model_path)
# This module is for evaluating GNN models
def evaluate_GNN(model,dataloader):
    # Training loop
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_num=1
        model=model.cuda(gpu_num)
    pred_acc = 0
    loss_va = 0
    counter = 0
    if "HGT" in str(type(model)):
        data_type="hetero"
    else:
        data_type="homo"
    for batch_graphs ,batch_labels,nodes,y_class,pred_state in dataloader:

        # Convert batch of DGL graphs to PyTorch Geometric`````````````````````````````` Data objects
        data=batch_graphs
        if torch.cuda.is_available():
            data=data.cuda(gpu_num)
        # Convert batch of DGL graphs to PyTorch Geometric Data objects

        # Prepare features and labels
        labels = batch_labels
        logits = model_output(model,data,data_type)
        logits=logits.cpu()

        loss = F.cross_entropy(logits, labels)

        # Calculate accuracy
        predicted_labels = torch.argmax(logits, dim=1)  # Get the class with the highest probability
        true_labels = torch.argmax(labels, dim=1)      # Get the true class labels
        correct_predictions = (predicted_labels == true_labels).sum().item()
        pred_acc += correct_predictions
        
        acc = correct_predictions / len(labels)
        #loss_his.append(loss.item())
        #acc_his.append(acc)

        # Backpropagation and optimization

        loss_va += loss.item() * len(labels)
        counter += len(labels)

    loss_va = loss_va / counter
    accuracy = pred_acc / counter

    # Print the loss and accuracy for each epoch
    return(accuracy * 100)
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, 10)  # Assuming input images are 28x28. Adjust if different.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.pool(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    def get_activation_functions(self):
        activation_functions = {}
        activation_functions['conv1'] = "None"
        activation_functions['pool'] = nn.ReLU()
        activation_functions['conv2'] = nn.ReLU()
        activation_functions['pool_1'] = 'None'  
        activation_functions['flatten'] = 'None' 
        activation_functions['fc'] = 'None'  
        activation_functions['softmax'] = 'None'  

        return activation_functions
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, CaptumExplainer
from typing import List, Union, Dict


def train_cifar10_pytorch(path,epochs=20):
    '''train a CIFAR10 pytorch model'''
    
    # Create a training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the model
    model = Cifar10_Net()
    model.to(device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Training code
    train_loader, test_loader = get_dataset('cifar10',model_type='pytorch')
    print('Training ...')
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch + 1}/epochs], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # evaluate model
    evaluate_model(model, test_loader,device=device)

    # save the model
    if path is not None:
        torch.save(model.state_dict(), path)



def explainer_GNN(model, mode: str, edge_mask_type: Union[str, None] = "object"):
    """
    mode: a Captum algorithm name accepted by torch_geometric.explain.CaptumExplainer,
          e.g., 'integrated_gradients', 'saliency', 'input_x_gradient', 'gradient_shap', etc.
    edge_mask_type: 'object' (mask per edge), 'scalar', or None to disable edge masks.
    """
    # If your model returns logits (recommended), keep return_type='logits'.
    # If your model returns probabilities, change to return_type='probs'.
    return Explainer(
        model=model,
        algorithm=CaptumExplainer(mode),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type=edge_mask_type,
        model_config=dict(
            mode="binary_classification",   # or 'multiclass_classification'
            task_level="graph",             # graph-level task
            return_type="probs",           # <-- change to 'probs' if your model returns probabilities
        ),
    )

@torch.no_grad()
def _num_graphs_from_hetero(b: HeteroData) -> int:
    # Use any node type's batch vector to infer number of graphs
    for nt in b.node_types:
        if hasattr(b[nt], "batch") and b[nt].batch is not None:
            return int(b[nt].batch.max().item()) + 1
    # Fallback if no batch attr found (treat as 1 big graph)
    return 1
def _class_index(label_batch: torch.Tensor, idx: int) -> int:
    """Return scalar class index for graph `idx` from various label shapes."""
    t = label_batch[idx]
    if t.ndim == 0:                 # scalar
        return int(t.item())
    if t.ndim == 1:                 # e.g., [C] one-hot or [1]
        if t.numel() == 1:
            return int(t.item())
        return int(t.argmax(dim=0).item())
    # fallback: flatten then argmax
    return int(t.view(-1).argmax(dim=0).item())
from typing import List, Union, Dict
def get_nodes_data_g(data_loader, model, mode: str, conv_exis: bool = False):
    """
    Generates one explanation per graph in each batch.

    data_loader should yield (batch, label, *_) where:
      - batch is Data or HeteroData (already collated by a PyG loader),
      - label is shape [B] (graph labels).

    conv_exis=True  -> heterogeneous (HeteroData)
    conv_exis=False -> homogeneous (Data)
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()  # Captum will still compute input grads; we just disable dropout, etc.

    samples_explanations: List = []
    global batch

    for batche, label, *rest in tqdm(data_loader):
        # Move batch & labels to device
        if isinstance(batche, (Data, HeteroData)):
            batche = batche.to(device)
        else:
            # If your loader wraps differently, ensure batch.to(device) is called
            pass
        label = label.to(device)

        if conv_exis:  # --- Heterogeneous graph path ---
            # Build inputs for Explainer
            x_dict: Dict[str, torch.Tensor] = {nt: batche[nt].x.detach().float().clone().requires_grad_(True)
                                               for nt in batche.node_types}
            edge_index_dict = {et: batche[et].edge_index for et in batche.edge_types}
            batch_dict = {nt: batche[nt].batch for nt in batche.node_types}

            # Build the explainer (no edge mask -> node attributions only) or keep object masks
            expl = explainer_GNN(model, mode, edge_mask_type=None)  # set to "object" if you want edge masks

            B = _num_graphs_from_hetero(batche)
            # Explain each graph in the hetero batch
            for idx in range(B):
                tgt = _class_index(label, idx)
                # IMPORTANT: pass batch_dict as a keyword to match model.forward signature
                exp = expl(
                    x_dict,
                    edge_index_dict,
                    batch_dict=batch_dict,
                    index=idx,                # which graph in the batch
                    target=tgt,              # class index for attribution
                )
                samples_explanations.append(exp.cpu())

        else:          # --- Homogeneous graph path ---
            explainer=explainer_GNN(model,mode)
            batch=batche.batch
            x=batche.x
            edge_index=batche.edge_index
            hetero_explanation = explainer(
            x,
            edge_index,target=label,index=0)
            hetero_explanation.cpu()
            samples_explanations.append(hetero_explanation)

    return samples_explanations
