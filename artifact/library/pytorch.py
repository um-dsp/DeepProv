
### Update get_dataset() in utils.py with a pytorch implementation:

def get_dataset(dataset_name, categorical=False, model_type='keras'):
    

    if(dataset_name == "mnist"):
        if model_type=='pytorch':
            # Define a transform to normalize the data
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            # Download and load the training data
            trainset = datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)
            train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

            # Download and load the test data
            testset = datasets.MNIST(root='./mnist', train=False, download=True, transform=transform)
            test_loader = DataLoader(testset, batch_size=64, shuffle=False)
        elif model_type=='keras':
            (X_train, Y_train), (X_test, Y_test)  = mnist.load_data()
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            X_train = X_train / 255.0
            X_test = X_test/ 255.0

    if(dataset_name == "cifar10"):
        if model_type=='keras':
            (X_train, Y_train), (X_test, Y_test)  = cifar10.load_data()
            X_train = X_train.reshape((X_train.shape[0],32,32,3))
            X_test = X_test.reshape((X_test.shape[0],32,32,3))
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            X_train = X_train / 255.0
            X_test = X_test/ 255.0
        if model_type=='pytorch':
            # Transformations - Convert images to PyTorch tensors and normalize them
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            # Load the training and test sets
            trainset = datasets.CIFAR10(root='./cifar10', train=True,
                                                    download=True, transform=transform)
            train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

            testset = datasets.CIFAR10(root='./cifar10', train=False,
                                                download=True, transform=transform)
            test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    if(dataset_name =="cuckoo"):
        df = pd.read_csv("./data/cuckoo.csv",encoding='iso-8859-1')
        df.drop(['Samples'],axis=1,inplace=True) # dropping the name of each row
        df_train=df.drop(['Target'],axis=1)
        df_train.fillna(0)
        df['Target'].fillna('Benign',inplace = True)    
        X= df_train.values
        Y=df['Target'].values
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.33, random_state=7)
    if(dataset_name == 'ember'):
        X_train, Y_train, X_test, Y_test = ember.read_vectorized_features("./data/ember2018/")
    
    if categorical:
        if(dataset_name == 'cuckoo'):
            Y_train = pd.get_dummies(Y_train)
            Y_test = pd.get_dummies(Y_test)
        else : 
            Y_train= to_categorical(Y_train)
            Y_test= to_categorical(Y_test)
    if(model_type=='keras'):
        return(X_train, Y_train), (X_test, Y_test)
    elif (model_type=='pytorch'):
        return train_loader, test_loader
    
        
        
        
        
### Add a pytorch implementation of the same Mnist model in train.py
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

def train_mnist_pytorch(path=None,epochs=15):

    # Create a training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    model = NeuralMnist()
    model.to(device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Training code
    train_loader, test_loader = get_dataset('mnist',model_type='pytorch')
    print('Training code')
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
    accuracy = evaluate_model(model, test_loader,device=device)
    print(f'Accuracy of mnist model on the test images: {accuracy}%')

    # save the model
    if path is not None:
        torch.save(model.state_dict(), path)

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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


### Add a pytorch implementation of the same Cifar10 model in train.py
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)

        # Conv Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2(x)

        # Flatten and Fully Connected Layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def train_cifar10_pytorch(path,epochs=20):
    '''train a CIFAR10 pytorch model'''
    
    # Create a training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    accuracy = evaluate_model(model, test_loader,device=device)
    print(f'Accuracy of cifar10 model on the test images: {accuracy}%')

    # save the model
    if path is not None:
        torch.save(model.state_dict(), path)
        
        
        
### You can test the code with:

# train mnist
from library.train import train_mnist_pytorch
train_mnist_pytorch(path='./models/mnist_1.pth')

# train cifar10
from library.train import train_cifar10_pytorch
train_cifar10_pytorch(path='./models/cifar10_1.pth')

## Accuracy mnist=97.69
## Accuracy cifar10=81.