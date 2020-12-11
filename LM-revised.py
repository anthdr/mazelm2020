#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from random import shuffle
import torch.optim as optim
import numpy as np


# In[2]:


#créer les items du vocabulaire: liste des mot, attribution d'un chiffre pour chacun des mots, et inverse
def vocabulary(filename,padding='<pad>',unknown='<unk>'):
    istream = open(str(filename), encoding="utf8")
    dict = []
    symbol = []
    for line in istream:
        if line and not line.isspace():
            symbol = line.split()
            dict.append(symbol)
    #flatten list of list
    dict = [item for sublist in dict for item in sublist]
    #make each string of list unique
    dict = list(set(dict))
    if padding is not None:
      dict.append(padding)
    if unknown is not None:
      dict.append(unknown)
    word2int = { sym : idx for (idx,sym) in enumerate(dict) }
    dict = {v: k for k, v in word2int.items()}
    dict = list(dict.values())
    print("Vocabulary size = %d"%(len(dict),))
    int2word = {v: k for k, v in word2int.items()}
    
    data = np.array(dict)
    np.savez("dict", data)
    
    data = np.array(word2int)
    np.savez("word2int", data)
    
    data = np.array(int2word)
    np.savez("int2word", data)
    
    return dict,word2int,int2word

def pad(sequence,pad_size,pad_token):
    #returns a list with additional pad tokens if needed
    return sequence + [pad_token]*( pad_size-len(sequence) )

def code_sequence(sequence,coding_map,unk_token=None):
    #takes a list of strings and returns a list of integers
    #return [ coding_map[word] for word in sequence if word in coding_map else word == unk_token]
    list = []
    for word in sequence:
        if word not in coding_map:
            word = coding_map[unk_token]
        else:
            word = coding_map[word]
        list.append(word)
    return list

def decode_sequence(sequence,decoding_map):
    #takes a list of integers and returns a list of strings 
    return [ decoding_map[word] for word in sequence]

def read_tokens(filename):
    #make every line as a list of string
    corpus = open(str(filename), 'r', encoding="utf8")
    tokens = []
    for line in corpus:
        if line and not line.isspace():
            symbol = []
            symbol = line.split()
            tokens.append(symbol)
    
    #retire les chapitres et sous chapitre (=), renvoie une liste aplatie
    tokenz = []
    for list in tokens:
        for i in range(len(list)):
            if list[0] != '=':
                tokenz.append(list[i])
    
    #Chaque phrase se retrouve dans une liste unique , cette délimitation se fait avec "."
    #renvoie une liste de liste
    tokens = []
    current_tokens = []
    for i in tokenz:
        if i == '.':
            #print('samurai')
            current_tokens.append('.')
            tokens.append(current_tokens)
            current_tokens = []
        else:
            current_tokens.append(i)
        
    return tokens

#renvoie le premier mot de la phrase + n-1 nombre de mot dans la phrase (pour enlever le ".")
def seqsplit(sequence):
    sentences = []
    for i in range(len(sequence)-2):
        #print('X is ',sequence[0:2+i])
        sentences.append(sequence[0:2+i])
    return sentences

#pour chaque phrase, renvoie le dernier mot de la phrase et les mots le précédant
def Xandy(sequence):
    X = []
    y = []
    #for list in sequence:
    for i in range(len(sequence)):
        X.append(sequence[i][0:1+i])
        y.append([sequence[i][-1]])
    return X,y

#transforme toutes les phrases en corpus en phrase X pour target y
def eachSentence(tokens):
    seqX = []
    seqy = []
    for list in tokens:
        sentences = seqsplit(list)
        seqX_current, seqy_current = Xandy(sentences)
        for i in seqX_current:
            seqX.append(i)
        for i in seqy_current:
            seqy.append(i)
    return seqX,seqy


# In[3]:



class DataGenerator:

        #Reuse all relevant helper functions defined above to solve the problems
        def __init__(self,filename, parentgenerator = None, pad_token='<pad>',unk_token='<unk>'):

              if parentgenerator is not None: #Reuse the encodings of the parent if specified
                  self.pad_token      = parentgenerator.pad_token
                  self.unk_token      = parentgenerator.unk_token
                  self.input_sym2idx  = parentgenerator.input_sym2idx 
                  self.input_idx2sym  = parentgenerator.input_idx2sym 
                  #self.output_sym2idx = parentgenerator.output_sym2idx 
                  #self.output_idx2sym = parentgenerator.output_idx2sym  
              else:                           #Creates new encodings
                  self.pad_token = pad_token
                  self.unk_token = unk_token
                  #TODO : Create 4 encoding maps from datafile 
                  #self.input_idx2sym,self.input_sym2idx   = ...
                  #self.output_idx2sym,self.output_sym2idx = ...
                  pass
                  #######################
                  self.input_idx2sym,self.input_sym2idx,self.input_int2word  = vocabulary(filename,padding=pad_token,unknown=unk_token)
                  #self.output_idx2sym,self.output_sym2idx = vocabulary(conllfilename,False,padding=pad_token,unknown=None)
                  #######################
                    
              #TODO : store the conll dataset with sentence structure (a list of lists of strings) in the following fields 
              #self.Xtokens = ...
              #self.Ytokens = ...
              pass
              ##########################
              self.sentences = read_tokens(filename)
              #print('tokens size is: ', len(self.Xtokens))
              #self.Ytokens = read_conll_tags(conllfilename)
              ##########################

        def generate_batches(self,batch_size,keep_order=False):

            N = len(self.sentences)
            idxes = list(range(N))
            sidxes = list(range(0,N,batch_size)) #batch start idxes
            if not keep_order:
                shuffle(idxes)
                idxes.sort(key=lambda x:len(self.sentences[x]))
                shuffle(sidxes)

            for sidx in sidxes:
               eidx = min(sidx+batch_size,N)
               batchlen = max( [ len(self.sentences[idx])-1 for idx in idxes[sidx:eidx] ] )
               X = [self.sentences[idx][:-1] for idx in idxes[sidx:eidx] ]
               Y = [self.sentences[idx][1:] for idx in idxes[sidx:eidx] ]
               print('X :',X)
               #print('\n \n \n')
               print('Y :',Y)
               #padding
               X = [pad(x,batchlen,pad_token = self.pad_token) for x in X]
               Y = [pad(y,batchlen,pad_token = self.pad_token) for y in Y]
               #coding
               xinput  = [ [self.input_sym2idx.get(token,self.input_sym2idx[self.unk_token]) for token in x] for x in X]
               youtput = [ [self.input_sym2idx.get(token,self.input_sym2idx[self.unk_token]) for token in y] for y in Y]
               yield (xinput,youtput)


# In[4]:



class NERtagger(nn.Module):

      def __init__(self,traingenerator, embedding_size,hidden_size,device='cuda'):
        super(NERtagger, self).__init__()        
        self.embedding_size    = embedding_size
        self.hidden_size       = hidden_size
        self.allocate_params(traingenerator,device) 

      def load(self,filename):
        self.load_state_dict(torch.load(filename))

      def allocate_params(self,datagenerator,device):
        
        pass #create fields for nn Layers

        #########################
        invocab_size    = len(datagenerator.input_idx2sym)
        outvocab_size   = len(datagenerator.input_idx2sym)
        pad_index       = datagenerator.input_sym2idx[datagenerator.pad_token]
        self.embeddings = nn.Embedding(invocab_size,self.embedding_size,padding_idx=pad_index).to(device)
        #self.lstm       = nn.LSTM(self.embedding_size,self.hidden_size).to(device)
        #self.sequential = nn.Sequential(self.embedding_size,self.hidden_size).to(device)
        #self.RNN        = nn.RNN(self.embedding_size,self.hidden_size).to(device)
        self.linear      = nn.Linear(self.embedding_size,self.hidden_size).to(device)
        self.linear_out = nn.Linear(self.hidden_size,outvocab_size).to(device)
        #self.softmax_out = nn.softmax(self.hidden_size,outvocab_size).to(device)
        #########################

      def forward(self,Xinput):

        pass #prediction steps

        ############################
        E    = self.embeddings(Xinput)
        #H,_  = self.lstm(E)
        #H,_  = self.sequential(E)
        H  = self.linear(E)
        return self.linear_out(H)
        #return self.softmax_out(H)
        ############################
        
      def train(self,traingenerator,validgenerator,epochs,batch_size,device='cuda',learning_rate=0.001): 

        self.minloss = 10000000 #the min loss found so far on validation data
        pass

        ############################
        device       = torch.device(device)
        pad_index    = traingenerator.input_sym2idx[traingenerator.pad_token]
        loss_fnc     = nn.CrossEntropyLoss(ignore_index=pad_index)
        optimizer    = optim.Adam(self.parameters(),lr=learning_rate)

        for e in range(epochs):
            L = [ ]
            i = 0
            y = 1
            for (seqX,seqY) in traingenerator.generate_batches(batch_size):
                optimizer.zero_grad()
                X = torch.LongTensor(seqX).to(device)
                Y = torch.LongTensor(seqY).to(device)
                
                Yhat = self.forward(X)

                #batch_size,seq_len = Y.shape
                #Yhat = Yhat.view(batch_size*seq_len,-1)
                #Y    = Y.view(batch_size*seq_len)
                #Yhat = Yhat.view(batch_size, len(traingenerator.input_idx2sym))
                print(' \n ')
                print('X is: ',X.shape)
                print('Y is: ',Y.shape)
                print('Yhat: ',Yhat.shape)
                bsize,seq_len = Y.shape
                Yhat = Yhat.view(bsize * seq_len, -1)
                Y    = Y.view(bsize*seq_len)
                print('X is: ',X.shape)
                print('Y is: ',Y.shape)
                #print(Y)
                print('Yhat: ',Yhat.shape)
                #print(Yhat)
                loss = loss_fnc(Yhat,Y)
                loss.backward()
                optimizer.step()
                L.append(loss.item())
                i =  i + batch_size
                print('batch :',i, ' ; epoch :',y)
            
            y = y + 1
            print(y)
            train_loss = sum(L)/len(L)
            print('Epoch %d'%(e+1,)) 
            print('[train] mean Loss = %f'%(train_loss,) )

            self.validate(validgenerator,batch_size,device=device,save_min_model=True)
            print()

        self.load('tagger_params.pt')
        #############################

      def validate(self,datagenerator,batch_size,device='cuda',save_min_model=False):
          
          batch_accurracies = []
          batch_losses      = []

          device = torch.device(device)
          pad_index = datagenerator.input_sym2idx[datagenerator.pad_token]
          loss_fnc  = nn.CrossEntropyLoss(ignore_index=pad_index)

          for (seqX,seqY) in datagenerator.generate_batches(batch_size):
                with torch.no_grad():   
                  X = torch.LongTensor(seqX).to(device)
                  Y = torch.LongTensor(seqY).to(device)
                
                  Yhat = self.forward(X)

                  #Flattening and loss computation
                  #batch_size,seq_len = Y.shape
                  #Yhat = Yhat.view(batch_size*seq_len,-1)
                  #Y    = Y.view(batch_size*seq_len)
                  bsize,seq_len = Y.shape
                  Yhat = Yhat.view(bsize * seq_len, -1)
                  Y    = Y.view(bsize*seq_len)
                  loss = loss_fnc(Yhat,Y)
                  batch_losses.append(loss.item())

                  #Accurracy computation
                  mask    = (Y != pad_index)
                  Yargmax = torch.argmax(Yhat,dim=1)
                  correct = torch.sum((Yargmax == Y) * mask)
                  total   = torch.sum(mask)
                  batch_accurracies.append(float(correct)/float(total))

          L = len(batch_losses)                  
          valid_loss = sum(batch_losses)/L

          if save_min_model and valid_loss < self.minloss:
            self.minloss = valid_loss
            torch.save(self.state_dict(), 'tagger_params.pt')

          print('[valid] mean Loss = %f | mean accurracy = %f'%(valid_loss,sum(batch_accurracies)/L))


# In[5]:


trainset = DataGenerator('wiki.train.raw')
validset = DataGenerator('wiki.test.raw',parentgenerator = trainset)
tagger   = NERtagger(trainset,4,4)
tagger.train(trainset,validset,2,8,device='cuda')


# In[ ]:




