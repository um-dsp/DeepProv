import torch.nn as nn
import torch.nn.functional as F
import torch

class Ember_model(nn.Module):
    def __init__(self,specific_indices,distribution_ben,alpha,beta):
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
        self.nodes_indices=specific_indices
        self.ben_distr=distribution_ben
        self.alpha=alpha
        self.beta=beta
    def act_on(self,x,specific_indices,distribution_ben,index=0):
        mask = torch.zeros_like(x , dtype=torch.bool,device=x.device)
        mask[:, specific_indices] = True
        x_new = torch.where(mask, distribution_ben, x)
        Delta=x-x_new
        mask=mask.cpu() & (abs(Delta).cpu()<self.beta[index]).cpu()
        mask=mask.to(x.device)
        x = x.clone()
        x[mask]=x[mask]-torch.mul(self.alpha, Delta)[mask]

        return(x)
    def forward(self, x):
        x = F.relu((self.input_layer(x)))
        specific_indices_0=self.nodes_indices[0]
        if len(specific_indices_0)>0:
            x=self.act_on(x,specific_indices_0,self.ben_distr[0],0)

        x=self.hidden_layer1(x)
        specific_indices_1=self.nodes_indices[1]
        if len(specific_indices_1)>0:
            x=self.act_on(x,specific_indices_1,self.ben_distr[1],1)
        x = F.relu(self.bn2(x))
        specific_indices_2=self.nodes_indices[2]
        if len(specific_indices_2)>0:
            x=self.act_on(x,specific_indices_2,self.ben_distr[2],2)
        x = self.dropout1(x)
        specific_indices_3=self.nodes_indices[3]
        if len(specific_indices_3)>0:
            x=self.act_on(x,specific_indices_3,self.ben_distr[3],3)
        x=self.hidden_layer2(x)
        specific_indices_4=self.nodes_indices[4]
        if len(specific_indices_4)>0:
            x=self.act_on(x,specific_indices_4,self.ben_distr[4],4)
        x = F.relu(self.bn3(x))
        specific_indices_5=self.nodes_indices[5]
        if len(specific_indices_5)>0:
            x=self.act_on(x,specific_indices_5,self.ben_distr[5],5)
        x = self.dropout2(x)
        specific_indices_6=self.nodes_indices[6]
        if len(specific_indices_6)>0:
            x=self.act_on(x,specific_indices_6,self.ben_distr[6],6)
        x=self.hidden_layer3(x)
        specific_indices_7=self.nodes_indices[7]
        if len(specific_indices_7)>0:
            x=self.act_on(x,specific_indices_7,self.ben_distr[7],7)
        x = F.relu(self.bn4(x))
        specific_indices_8=self.nodes_indices[8]
        if len(specific_indices_8)>0:
            x=self.act_on(x,specific_indices_8,self.ben_distr[8],8)
        x = self.dropout3(x)
        specific_indices_9=self.nodes_indices[9]
        if len(specific_indices_9)>0:
            x=self.act_on(x,specific_indices_9,self.ben_distr[9],9)
        x = torch.sigmoid(self.output_layer(x))
        return x
class NeuralMnist_v1(nn.Module):

    '''PyTorch Implementation of Mnist model'''

    def __init__(self,specific_indices,distribution_ben,alpha,beta):
        super(NeuralMnist_v1, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 350)
        self.fc2 = nn.Linear(350, 50)
        self.fc3 = nn.Linear(50, 10)
        self.nodes_indices=specific_indices
        self.ben_distr=distribution_ben
        self.alpha=alpha
        self.beta=beta
    def act_on(self,x,specific_indices,distribution_ben,index=0):
        mask = torch.zeros_like(x , dtype=torch.bool,device=x.device)
        mask[:, specific_indices] = True
        x_new = torch.where(mask, distribution_ben, x)
        Delta=x-x_new
        mask=mask.cpu() & (abs(Delta).cpu()<self.beta[index]).cpu()
        mask=mask.to(x.device)
        x = x.clone()
        x[mask]=x[mask]-torch.mul(self.alpha, Delta)[mask]

        return(x)

    def forward(self, x):
        x = self.flatten(x)
        specific_indices_0=self.nodes_indices[0]
        if len(specific_indices_0)>0:
            x=self.act_on(x,specific_indices_0,self.ben_distr[0],0)
        #x = torch.where(x<=-0.7, -1, x)
        x = F.relu(self.fc1(x))
        specific_indices_1=self.nodes_indices[1]
        if len(specific_indices_1)>0:
            x=self.act_on(x,specific_indices_1,self.ben_distr[1],1)
        x = F.relu(self.fc2(x))
        specific_indices_2=self.nodes_indices[2]
        if len(specific_indices_2)>0:
            x=self.act_on(x,specific_indices_2,self.ben_distr[2],2)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class MLPClassifierPyTorch_V1(nn.Module):
    def __init__(self,specific_indices,distribution_ben,alpha,beta):
        super(MLPClassifierPyTorch_V1, self).__init__()
        self.fc1 = nn.Linear(in_features=1549, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=18)
        self.fc3 = nn.Linear(in_features=18, out_features=12)
        self.fc4 = nn.Linear(in_features=12, out_features=1)
        self.nodes_indices=specific_indices
        self.ben_distr=distribution_ben
        self.alpha=alpha
        self.beta=beta
    def act_on(self,x,specific_indices,distribution_ben,index=0):
        mask = torch.zeros_like(x , dtype=torch.bool,device=x.device)
        mask[:, specific_indices] = True
        x_new = torch.where(mask, distribution_ben, x)
        Delta=x-x_new
        mask=mask.cpu() & (abs(Delta).cpu()<self.beta[index]).cpu()
        mask=mask.to(x.device)
        x = x.clone()
        x[mask]=x[mask]-torch.mul(self.alpha, Delta)[mask]

        return(x)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        specific_indices_0=self.nodes_indices[0]
        if len(specific_indices_0)>0:
            x=self.act_on(x,specific_indices_0,self.ben_distr[0],0)
        x = F.relu(self.fc2(x))
        specific_indices_1=self.nodes_indices[1]
        if len(specific_indices_1)>0:
            x=self.act_on(x,specific_indices_1,self.ben_distr[1],1)
        x = F.relu(self.fc3(x))
        specific_indices_2=self.nodes_indices[2]
        if len(specific_indices_2)>0:
            x=self.act_on(x,specific_indices_2,self.ben_distr[2],2)
        x = self.fc4(x)  # No activation in the last layer for binary classification
        return F.sigmoid(x)
    
