import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM
#auto tokenizer and automodel are for bert-base-historical-german-cased from @redewiedergabe
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from utils import printProgressBar
from sklearn.metrics import f1_score, recall_score, confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle #to save the model into a file .pkl 
 

""" 
Change the tokenizer if you want to test mybert or historicalBert 
"""
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
#tokenizer = AutoTokenizer.from_pretrained("redewiedergabe/bert-base-historical-german-rw-cased")

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = df['label'].values
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-german-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 3)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        #_ contains the embedding vectors of all of the tokens in a sequence
        # pooled_output contains the embedding vector of [CLS] token. 
        # For a text classification task, it is enough to use this embedding as an input for our classifier
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
    

    
def train(model, train_data, val_data, learning_rate, epochs, label, hist):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                if hist==True:
                    """ Only for bert-historical from pretrained"""
                    output=output.logits

                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    if hist==True:
                        """ Only for bert-historical from pretrained"""
                        output=output.logits


                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
                    
            
            with open(label, 'wb') as fid:
                pickle.dump(model, fid)  
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}')


def evaluate(model, test_data, hist):

    counts=test_data['label'].value_counts()
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    total_acc_class=[0,0,0]
    y_true, y_pred= [],[]
    
    l = len(test_data)/2
    i=0
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            
              flag=0
              
              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)
              
              y_true.append(test_label.data[0].item())

              try:
                y_true.append(test_label.data[1].item())
              except:
                    flag=1
                    print("batch of size 1")
              #It is for bert-base-german-case (mybert)
              if hist==False:
                
                y_pred.append(output.argmax(dim=1)[0].item())

                if flag==0:
                    y_pred.append(output.argmax(dim=1)[1].item()) 

                
                
                if(output.argmax(dim=1)[0] == test_label[0]):
                    total_acc_class[int(test_label.data[0].item())]+=1
                    
                if flag ==0:    
                    if(output.argmax(dim=1)[1] == test_label[1]):
                        total_acc_class[int(test_label.data[1].item())]+=1
                
                
                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc
             
              
              #For bert historical (historicalbert)
              if hist == True:
                y_pred.append( np.argmax(output.logits[0]))
                y_pred.append(np.argmax(output.logits[1]))
                if(np.argmax(output.logits[0]) == test_label[0]):
                        
                        total_acc_class[int(test_label.data[0].item())]+=1
                        
                if(np.argmax(output.logits[1]) == test_label[1]):

                        total_acc_class[int(test_label.data[1].item())]+=1
                
                
                acc = (np.argmax(output.logits) == test_label).sum().item()
                total_acc_test += acc

               #end bert historical
              
              # Update Progress Bar
              printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
              i+=1
    
    print("all values: ", total_acc_class)
    print("Accuracy per Class: ")
    print(f"Class 0 Nein -> { ( total_acc_class[0]/counts[0] if total_acc_class[0]>0 else 0  ): .3f} on {counts[0]} samples")
    print(f"Class 1 Metapher ->  { total_acc_class[1]/counts[1]: .3f} on {counts[1]} samples")
    print(f"Class 2 Kandidat ->  { total_acc_class[2]/counts[2]: .3f} on {counts[2]} samples")
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f} \n\n')
    
    print("F1-score: ")
    score= f1_score(y_true, y_pred, average=None)
    print(score)
    print("Class 0 Nein score-> ", score[0])
    print("Class 1 Metapher score -> ", score[1])
    print("Class 2 Kandidat score -> ", score[2])
    
    print("F1 wih average Macro -> ",f1_score(y_true, y_pred, average='macro'),"\n" )
    
    rec=recall_score(y_true, y_pred, average=None)
    print("Class 0 Nein recall-> ", rec[0])
    print("Class 1 Metapher recall -> ", rec[1])
    print("Class 2 Kandidat recall -> ", rec[2])
    print("Recall score macro -> ", recall_score(y_true, y_pred, average='macro'),"\n")
    
    
    print("Confusion Matrix: \n")
    cfm= confusion_matrix(y_true, y_pred)
    print(cfm)
    df_cm = pd.DataFrame(cfm, range(3), range(3))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True) # font size
    plt.show()
    