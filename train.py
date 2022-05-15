import json
from nltk_ultil import tokenize,stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json','r') as f:
    intents=json.load(f)

all_word=[]
tags=[]
#holds both patterns nd tags
xy=[]

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        #tokenize pattern
        w=tokenize(pattern)
        #use extend cuzit an array not arr[arr]
        all_word.extend(w)
        xy.append((w,tag))

ignore_words=['?','!','.',',']
#to remove ignore word of list
all_word=[stem(w) for w in all_word if w not in ignore_words]
all_word=sorted(set(all_word))
tags=sorted(set(tags))
#print(tags)

#now create traing data

X_train=[]
y_train=[]
for(pattern_sentence,tag) in xy:
    bag=bag_of_words(pattern_sentence,all_word)
    X_train.append(bag)

    labels=tags.index(tag)
    y_train.append(labels) #crossentropy loss

X_train=np.array(X_train)
y_train=np.array(y_train)


#create pytorch datset 4 batch train
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples=len(X_train)
        self.x_data=X_train
        self.y_data=y_train
    
    #data[index]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
    
    
batch_size=8
hidden_size=8
output_size=len(tags)
input_size=len(X_train[0])
learning_rate=0.001
num_epochs=1000
"""
print(input_size,len(all_word))
print(output_size,tags)
"""

dataset=ChatDataset()
train_loader=DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model= NeuralNet(input_size,hidden_size,output_size).to(device)

#loss & optiimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop

for epochs in range(num_epochs):
    for (words, labels) in train_loader:
        words=words.to(device)
        labels=labels.to(device)

        #forward
        labels=labels.long()
        outputs=model(words)
        loss=criterion(outputs,labels)


        #backwards & optimixer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epochs+1)%100==0:
        print(f'epoch {epochs+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

#save the data
data={
    'model_state':model.state_dict(),
    'input_size':input_size,
    'output_size':output_size,
    'hidden_size':hidden_size,
    'all_words':all_word,
    'tags':tags

}
FILE="data.pth"
torch.save(data,FILE)
print("training complete!, file saved!!")
