from operator import contains
from library.Activations import Activations
import os
import pandas as pd
import numpy as np
from keras import backend as K
from library.utils import *
class Accessor :

    # for all class attack param selects whether to get adersarial activations or begnign activations


    def __init__(self,folder):
        self.folder = folder
    
   
    #get activations by label and prediction [get activations of an input predicted 0 and labeled 0]
    def get_instance_by_label_prediction(self,label,prediction):        
        for filename in os.listdir(self.folder):
            f = os.path.join(self.folder, filename)
            # checking if it is a file
            if filename.startswith(str(label) +"_"+ str(prediction)):
                index = filename[filename.index("-")+1:filename.index(".")] 
                activations_set = self.parse_csv_to_set(pd.read_csv(f))
                print('Loaded Activations of image labeled %s predicted %s ' % (label,prediction))
                return  Activations(index,label,prediction,activations_set)


    def get_label_by_prediction(self,target_prediction,verbose = 1,collapse='avg',limit = float('+inf')):
        
        container = []
        counter = 0
        for filename in os.listdir(self.folder):
            if(counter>limit):
                break
            f = os.path.join(self.folder, filename)
            predicted = filename[filename.index("_")+1:filename.index("-")] 
            if int(predicted) == target_prediction:
                index = filename[filename.index("-")+1:filename.index(".")] 
                if(filename.find('csv') != -1):
                    activations_set = self.parse_csv_to_set(pd.read_csv(f))
                else :
                    activations_set = self.parse_txt_to_set(f,collapse)

                label = filename[0:filename.index("_")] 
                container.append(Activations(index,label,predicted,activations_set))
                counter += 1
        if( len(container) ==0):
            print('No File was found for prediction %s'%(target_prediction))
            raise Exception()
        if(verbose ==1 ):
            print('Loaded '+str(len(container))+' Activations for Prediction : '+str(target_prediction))
        return container
    

  

    #gets activations for a given class [0,1,...]
    def get_by_label(self,label,collapse='avg',limit= float('+inf'),verbose=0):
        container = []
        counter = 0 
        for filename in os.listdir(self.folder):
            if(verbose):
                printProgressBar(counter, limit, prefix = 'Progress:', suffix = 'Complete', length = 50)
            if(counter >= limit):
                break
            f = os.path.join(self.folder, filename)
            if filename.startswith(str(label)):
                counter+=1
                index = filename[filename.index("-")+1:filename.index(".")] 
                predicted = filename[filename.index("_"):filename.index("-")] 
                if(filename.find('csv') != -1):
                        activations_set = self.parse_csv_to_set(pd.read_csv(f))
                else :
                        activations_set = self.parse_txt_to_set(f,collapse)
                container.append(Activations(index,label,predicted,activations_set))
        if( len(container) ==0):
            print('No File was found for label %s'%(label))
            raise Exception()
        print('Loaded '+str(len(container))+' Activations for Label : '+str(label))
        return container
    
    #gets activations by input index in dataset
    def get_instance_by_index(self,index):
        for filename in os.listdir(self.folder):
            f = os.path.join(self.folder, filename)
            i = int(filename[filename.index("-")+1:filename.index(".")] )
            if (index ==i) :
                predicted = filename[filename.index("_"):filename.index("-")] 
                label = filename[0:filename.index("_")] 
                activations_set = self.parse_csv_to_set(pd.read_csv(f))
                print("Loaded instance indexed %s labeled %s predicted %s  "%(i,label,predicted))
                return Activations(index,label,predicted,activations_set)
        raise Exception("No file found for index %s" ,index)


    def get_all(self,collapse='avg',sub_ration = 0,limit = float('+inf'),start=0):
        container = []
        current= 0
        for filename in os.listdir(self.folder):
            #current+=1
            if(current>limit): break
            if(current<start):continue
            #purpose of this is to shrink number of laoded activaitons to sub_ratio % (for faster loasd during research)
            #if( sub_ration != 0 and not current % sub_ration == 0):
            #    continue
            f = os.path.join(self.folder, filename)
            index = filename[filename.index("-")+1:filename.index(".")] 
            predicted = filename[filename.index("_"):filename.index("-")] 
            label = filename[0:filename.index("_")] 

            if(filename.find('csv') != -1):
                    activations_set = self.parse_csv_to_set(pd.read_csv(f))
            else :
                    activations_set = self.parse_txt_to_set(f,collapse)
            #for layer_set in activations_set:
            #    print(len(layer_set))
            container.append(Activations(index,label,predicted,activations_set))
            current+=1
        if( len(container) ==0):
            print('No File was found for label %s'%(label))
            raise Exception()
        print('Loaded all activations for %s' %(self.folder))
        
        return container
            


    #todo
    #implement get adv and begnign per sammple
    # IMPLEMENT GET smaller dataset containing 1,2,... or n classes only

    def parse_csv_to_set(self,file_content):
        set = []
        for col in file_content:
            set.append(file_content[col].dropna())
        
        return set
  
    def float_seq_from_line(l):
        return
    
    def parse_txt_to_set (self,file,collapse):
        activations_set = []
        extracted_nodes = 0
        with open(file) as f:
            lines = f.readlines()
            if(len(lines) ==0):
                raise Exception('Empty File')
            layer = -1
            for l in lines : 
                if(len(l)==0):
                    print('Empty line detected')
                    continue
                if('Layer' in l):
                    start = l.find(':')
                    end = l.find(';')
                    new_layer = int(l[start+1:end])
                    if new_layer<layer:
                        break
                        #raise Exception('Found an already loaded layer in file {} !'.format(file))
                    else:
                        layer=new_layer
                    ## Debugging code
                    #if layer not in [-1,0]:
                    #    print('{} nodes are extracted for layer {}'.format(extracted_nodes,layer-1))
                    #    for layer_set in activations_set:
                    #        print(len(layer_set))
                    activations_set.append([])
                    #print('extracting activations for layer',layer)
                    #extracted_nodes = 0
                #elif("Node" in l ) and layer != -1:
                #    start = l.find(':')
                #    end = l.find(';')
                    #node = int(l[start+1:end])
                    #activations_set[layer].append([])
                elif ('[' in l) and layer != -1:
                    s = np.fromstring(l.strip('[]'),sep=',',dtype='float32')
                    if(collapse == "avg"):
                        s = np.average(s)
                    #activations_set[layer][node]= s 
                    activations_set[layer].append(s) 
                    extracted_nodes+=1
            #if layer not in [-1,0]:
            #    print('{} nodes are extracted for layer {}'.format(extracted_nodes,layer))
            #    for layer_set in activations_set:
            #        print(len(layer_set))
            #print('Completed all lines')
                
        return activations_set
        

    
if __name__  == "__main__":
    a = Accessor("./Ground_truth/cifar10/cifar10_1")
    a= a.get_label_by_prediction(1,verbose =0,collapse='avg')
    for i in a[0].activations_set : 
        i = np.array(i)
        print(np.squeeze(i).shape)


