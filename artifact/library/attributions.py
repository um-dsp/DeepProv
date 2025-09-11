import torch
import torch.nn.functional as F
from library.Accessor import *
import numpy as np 
from sklearn.model_selection import train_test_split
#from library.attributionUtils import NeuralNetMnist_1,NeuralNetMnist_2,NeuralNetMnist_3, NeuralNetEmber, NeuralNetCuckoo_1, NeuralNetCifar_10
from library.attributionUtils import attribtuions_to_polarity,predict_torch,compute_accuracy_torch
from library.attributionUtils import get_attributes,adversarial_detection_set
from library.utils import normalize,dispersation_index
from library.Activations import Activations
from library.train import binary_acc
from sklearn.preprocessing import StandardScaler 





'''
    To categorical  :
    [1. 0.] => 0   =>Benign

'''




def get_average(X):
    if(len(X)==0) : raise ValueError('Cannot compute average of an empty array')
    return np.average(np.array(X)) /len(X)

def remove_elements_below_threshhold(attributes,x,theshhold):
    aux = []
    for i in attributes :
        if( abs(i)< abs(theshhold)):
            continue
    aux.append(i)
    print(len(aux))
    return torch.tensor(aux)

def normalize(set):
    #Normalize a set to range [-1,1]
    min_val = min(set)
    max_val = max(set)
    normalized_set = []
    for val in set:
        normalized_val = ((val - min_val) / (max_val - min_val)) * 2 - 1
        normalized_set.append(normalized_val)
    return normalized_set
def get_items_below_threshhold(attributes,threshhold):
    #given a set S this function returns ths iindices of elements that are below the threshholdin  abs value
    # this is for attribution to remove nodes who's attributions are ireelevant
    aux= []
    for index,x in enumerate(attributes):
        if(abs(x) < abs(threshhold)):
            continue
        aux.append(index)
    return aux

def remove_element_with_indexes(array,indexes):
    #given a set and an array of indexes this funciton remives the all the indexes form the array
    aux =[]
    for i,x in enumerate (array) : 
        if( i in indexes):
            continue
        aux.append(x)
    return torch.Tensor(aux)

def stats(attributes):
    pos =0
    neg= 0
    z = 0
    for i in attributes:
        if(i<0):neg+=1
        if(i>0):pos+=1
        else : z +=0
    
    return pos >neg

def multiply_attributed_with_input(X,Y,model):
   
    '''
    X: set of samples with the same label 0 or 1
    Y: is the set of labels
    
    '''
   
    s = []
    
    
    
    if torch.cuda.is_available():
        model=model.cuda()
    
    model.eval()
    X =torch.Tensor(X)
    Y=torch.Tensor(Y)
    y_pred = model(X)
    
    y_preds = torch.round(y_pred)
    #label = Y.unsqueeze(1)[0]
    #for index, x in enumerate(X):
       
    #prediction = y_preds[index]
    
    # filter to only correct prediction
    #if( prediction != label): continue
    
    attributes = get_attributes(X,model)#,label)           
    # find indexes of attributes that are below the thresshhold
    #indexes_to_remove = get_items_below_threshhold(attributes,0.01)
    
    #remove those indexes from attribtuions and from input { Considered as noise}
    #attributes = remove_element_with_indexes(attributes,indexes_to_remove)
    #input = remove_element_with_indexes(input,indexes_to_remove)

    #multiply  attributes with input
    
    mul = np.multiply(np.absolute(attributes),X)

    #s.append(mul)
    #printProgressBar(index, len(X), prefix = 'Progress:', suffix = 'Complete', length = 50)
    return mul, attributes

def number_of_active_nodes(set):
    threshold = 0.0001
    counter = 0
    for i in set : 
        if( i != 0 ):
            counter+=1
    return counter

