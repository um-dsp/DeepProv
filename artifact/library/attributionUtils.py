import torch
from captum.attr import IntegratedGradients
import torch.nn as nn
import random
import os
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import numpy as np 
import pickle
from library.train import get_nodes_data_g
from library.utils import dispersation_index,load_graph_data
from sklearn.model_selection import train_test_split

def attribtuions_to_polarity(X_test,y_test,model,target_label):
    
    positive = []
    negative = []
    index = 0 
    for sample_index in range(len(X_test)):
        sample = X_test[sample_index]

        label = int(y_test[sample_index][1].item())
        if(label != target_label):continue

    
        a = get_attributes (sample,model,label)

        pos = []
        neg = []

        for index,i in enumerate(a) : 
            if(i.item()>0):
                pos.append(index)
            if(i.item()<0):
                neg.append(index)


        positive.append(pos)
        negative.append(neg)
        index+=1
    return positive,negative

def get_attributes (data,model,model_name="",nbr_batch=0,mode="",task="",attack=None,folder="",conv_exis=False,root_folder="attributions_data/"):
    if task=="GNN_explainer" :
        save_path=root_folder+mode+"/"+model[1]
        if not (os.path.exists(save_path)):
                os.makedirs(save_path)
        if attack:
            id="_attributes_"+attack
        else:
            id="_attributes_ben"
        for nbr in range(nbr_batch):
            data_ben=load_graph_data(data,model_name,attack=attack,folder=folder,nbr_l_batches=nbr)
            data_ben=DataLoader(data_ben, batch_size=1, shuffle=False)
            samples_attributes=get_nodes_data_g(data_ben,model[0],mode,conv_exis)
            with open(save_path+'/batch_'+str(nbr)+id+'.pickle', 'wb') as handle:
                pickle.dump(samples_attributes, handle, protocol=pickle.HIGHEST_PROTOCOL)
            samples_attributes=[]
    else :
        #print(label.item())
        #print(data.shape)
        #sample = torch.reshape(sample,(1,1,sample.shape[0]))
        #print(sample.shape)
        ig = IntegratedGradients(model)
        attribution = ig.attribute(data, target=0)
        print(attribution.shape)
        #a= attribution[0][0]
        return attribution



    
def get_categoriztion_mapping(X,Y):
    X = torch.tensor(X)
    label_check =Y[0]
    Y= to_categorical(Y)
    cat_label = Y[0]
    Y = torch.tensor(Y)
    print(f'Label check {label_check} => {cat_label}')

def adversarial_detection_set(act,label):
    # Converts dlattned activations to a set :  X = [activation1,activation2,......,activation2] and Y [Begnign, Benign,....,Benign]
    X= []
    Y = []
    for x in act:
        #for layer in x.get_activations_set():
        #    print(len(layer))
        X.append(x.flatten())
        Y.append(label)

    return torch.tensor(X),torch.tensor(Y)


def randomize_tensor(tensor):
    return tensor[torch.randperm(len(tensor))]

def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors
    
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = torch.linalg.vector_norm(x - y, p, 2) if torch.__version__ >= '1.7.0' else torch.pow(x - y, p).sum(2)**(1/p)
    
    return dist
class NN():

    def __init__(self, X = None, Y = None, p = 2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        dist = distance_matrix(x, self.train_pts, self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):

    def __init__(self, X = None, Y = None, k = 3, p = 2):
        self.k = k
        super().__init__(X, Y, p)
    
    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        dist = distance_matrix(x, self.train_pts, self.p)

        knn = dist.topk(self.k, largest=False)
        votes = self.train_label[knn.indices]

        winner = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device)
        count = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device) - 1

        for lab in self.unique_labels:
            vote_count = (votes == lab).sum(1)
            who = vote_count >= count
            winner[who] = lab
            count[who] = vote_count[who]

        return winner

def box_plot(ben,ben_,adv,adv_,expected_nb_nodes):
    #banign / dav in benign samples
    ben,adv = get_nodes_weight_per_label(1)
    #banign / dav in adversarail samples
    adv_,ben_ = get_nodes_weight_per_label(0)
    
    case1 = 0 
    case2 = 0
    case3 = 0
    case4 = 0
    nb = 0 
    x = []
    while(nb <10):
        i = random.randint(1,expected_nb_nodes)
        all_ben = ben[i] + ben_[i]
        all_adv = adv[i] + adv_[i]
        if(not all_ben or not all_adv): continue
        print(f'Average ben : {np.average(all_ben)}  adv {np.average(all_adv)}')
        print(f'Range [ {min(all_ben)} :{max(all_ben)}]   adv {min(all_adv)} :{max(all_adv)} ')
        avg_ben = np.average(all_ben)
        avg_adv = np.average(all_adv)
        min_ben =min(all_ben)
        max_ben= max(all_ben)
        min_adv =min(all_adv)
        max_adv= max(all_adv)
        #x[ 'B' +str(i)] = all_ben
        #x['A'+str(i)] = all_adv
        nb+=1
        x.append(avg_adv - avg_ben)    
        if(max_ben <min_adv):
            case1 +=1
        elif(max_adv<min_ben):
            case2+=1
        elif(avg_ben<avg_adv):
            case3+=1
        elif(avg_ben>avg_adv):
            case4+=1

    fig, ax = plt.subplots()
    box = ax.boxplot(x.values(),patch_artist=True)
    for i,v in enumerate(box["boxes"]):
        if(i%2 == 0 ): continue
        plt.setp(v, color="red")

        print(i)
    ax.set_xticklabels(x.keys())

    plt.show()


def get_nodes_data(X,attributes):
    
    nb_nodes = X.shape[1]
    nodes_weights = [[] for i in range(nb_nodes)]
    nodes_avg_weights = []
    nodes_atts = [[] for i in range(nb_nodes)]
    nodes_avg_atts = []
    
    # Iterating the nodes of the target model
    for N in range(nb_nodes):
        # Iterating all samples
        for i,att in enumerate(attributes):
            # Store the weight (activation) of Node N corresponding to sample i
            nodes_weights[N].append(X[i][N])
            # Store the attribute of Node N corresponding to sample i
            nodes_atts[N].append(attributes[i][N])
        nodes_avg_weights.append(np.mean(nodes_weights[N]))
        nodes_avg_atts.append(np.mean(nodes_atts[N]))  
    return nodes_weights,nodes_atts, nodes_avg_weights, nodes_avg_atts

def get_avg_number_of_nodes_per_state(label,model,X,Y):
    pos = 0
    neg = 0
    counter = 0
    for index,input in enumerate(X):
        if(Y[index][1] != label) : continue
        attributes = get_attributes(input,model,label)
        for i,x in enumerate(attributes):
            if(x>0) : pos+=1
            if(x<0) : neg+=1
        counter+=1
    print(f'positive {pos} negative {neg}')
    return pos/counter, neg/counter

    

def predict_torch(model,x):
    x = torch.unsqueeze(x,dim=0)
    prediction = model(x)
    return torch.argmax(prediction)

def cluster(model,input,X,Y):
    clusteringX = []
    clusteringY = []
    for index,input in enumerate(X):
        label = int(Y[index][1].item())
        prediction = predict_torch(model,input).item()
        
        attributes = get_attributes(input,model,label)
        mul = np.multiply(attributes,input)
        clusteringX.append(np.array(mul))
        clusteringY.append(label)

        
    X_train, X_test,y_train, y_test = train_test_split(clusteringX,clusteringY ,random_state=104, test_size=0.25, shuffle=True)


    X_test = X_test[1:100]
    y_test = y_test[1:100]

    X_train = torch.FloatTensor(np.array(X_train))
    y_train = torch.LongTensor(np.array(y_train))
    X_test = torch.FloatTensor(np.array(X_test))

    knn = KNN(X_train, y_train)
    pred = knn(X_test)

    match =0
    for x, y in zip(pred,y_test):
        if(x==y):
            match+=1
        print(f'Accuracy : {match/len(X_test) * 100}')
def scatter(adv,ben):
    
    max_a = []
    max_b = []  
    min_a = []
    min_b = []
    disp_a = []
    disp_b = []
    avg_a = []
    avg_b = []

    for x,y in zip(adv,ben):
   
        x= np.array(x)
        y= np.array(y)

        max_a.append(max(x))
        max_b.append(max(y))
        min_a.append(min(x))
        min_b.append(min(y))
        disp_a.append(dispersation_index(x))
        disp_b.append(dispersation_index(y))
        avg_a.append(np.average(x))
        avg_b.append(np.average(y))

 
    figure, axis = plt.subplots(2, 2,figsize=(10, 100))
    figure.suptitle('Ember Emb attributes  analysis 500samples ')  

    axis[0, 0].scatter(np.arange(len(avg_a)),avg_a, s= 0.3,color="red")
    axis[0, 0].scatter( np.arange(len(avg_b)),avg_b, s= 0.3,color="green")
    axis[0, 0].title.set_text('Average ')


    axis[0, 1].scatter( np.arange(len(adv)),min_a, s= 0.3,color="red")
    axis[0, 1].scatter( np.arange(len(ben)),min_b, s= 0.3,color="green")
    axis[0, 1].title.set_text('Min Values')
    axis[0, 1].axhline(y = np.average(min_a), color = 'red', linestyle = '-')
    axis[0, 1].axhline(y = np.average(min_b), color = 'green', linestyle = '-')

    axis[1, 0].scatter(np.arange(len(adv)),max_a, s= 0.3,color="red")
    axis[1, 0].scatter( np.arange(len(ben)),max_b, s= 0.3,color="green")
    axis[1, 0].title.set_text('Max Values')
    axis[1, 0].axhline(y = np.average(max_a), color = 'red', linestyle = '-')
    axis[1, 0].axhline(y = np.average(max_b), color = 'green', linestyle = '-')

    plt.ylim(-1, 1)
    axis[1,1].scatter( np.arange(len(adv)),disp_a, s= 0.3,color ='red')
    axis[1,1].scatter( np.arange(len(ben)),disp_b, s= 0.3, color ="green")
    axis[1, 1].title.set_text(' Dispersation indexes')

    plt.show()

def get_nodes_impact_for_one_label(X_test, y_test, model, label,expected_nb_nodes):
    positive, negative = attribtuions_to_polarity(X_test,y_test,model,label)


    nodes_freq_p = [0 for i in range(expected_nb_nodes)]
    nodes_freq_n = [0 for i in range(expected_nb_nodes)]

    print(f'number of sampels studies {len(positive)}')
    for i in positive : 
        for j in i : 
            nodes_freq_p[j]+=1 / len(positive)

    for i in negative : 
        for j in i : 
            nodes_freq_n[j]+=1 / len(negative)


    always_pos = []
    always_neg = []
    both = []
    neutral = []
    for index,(i,j) in enumerate(zip(nodes_freq_p,nodes_freq_n)):
        if(i == 0 and j !=0 ) :
            always_pos.append(index)
        if(i != 0 and j ==0 ) :
            always_neg.append(index)
        if(i == 0 and j == 0 ) :
            neutral.append(index)
        if(i != 0 and j !=0 ) :
            both.append(index)

    print(f'For label {label}')
    print(f'Number of  always positive nodes : {len(always_pos)}')
    print(f'Number of  always negative nodes : {len(always_neg)}')
    print(f'Number of  neutral nodes : {len(neutral)}')
    print(f'Number of  undisicive nodes : {len(both)}')

    return always_pos,always_neg,neutral,both


def compute_accuracy_torch(X,Y,model):

    Y = to_categorical(Y,num_classes=2)
    Y = torch.tensor(Y)
    criterion = nn.CrossEntropyLoss()
    correct= 0
    index = 1
    for x, y in  zip(X,Y):  
          
            x = x[None, :]
            # Forward pass
            outputs = model(x)
            outputs = torch.squeeze(outputs)
            #print(torch.argmax(outputs),torch.argmax(y))

            loss = criterion(outputs, y)
            correct += 1 if (torch.argmax(outputs) == torch.argmax(y)) else 0 
            # Backward and optimize
            #print(f'accuracy {correct/index *100}')
            index+=1
    accuracy = correct/len(X) *100
    return accuracy