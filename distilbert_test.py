import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from transformers import Trainer, AutoModelForSequenceClassification, ConvBertForSequenceClassification
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')

from models import BertClassifier, train, Dataset, evaluate
import pickle #save and load the trained model


df= pd.read_csv(r"data/total_soft_012.csv")
df=df[['text','label']]
df.text.str.strip()

#80,10,10
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df)), int(.9*len(df))])

EPOCHS = 5
model = BertClassifier()
LR = 1e-6

""" 
# save the classifier
with open('mybert.pkl', 'wb') as fid:
    pickle.dump(model, fid)                 

print("Training: ")
train(model, df_train, df_val, LR, EPOCHS, 'mybert.pkl', hist=False) #we update the saved model after every epoch (see line 129-130 of models.py)
  
"""

with open('mybert.pkl', 'rb') as f:
    model = pickle.load(f)



# Trained with two epochs like the SoSe22 
#model = AutoModelForSequenceClassification.from_pretrained("redewiedergabe/bert-base-historical-german-rw-cased", num_labels=3)
#train(model, df_train,df_val,LR,EPOCHS,'berthistorical.pkl', hist=True)

""" with open('berthistorical.pkl', 'rb') as f:
    model = pickle.load(f)
 """
df_original=pd.read_csv(r"data/original_data1.csv")
df_original=df_original[['text','label']]
df_original.text.str.strip()

print("Evaluation: ")
evaluate(model, df_original, hist=False) #historical bert

""" 
Eval on df_test for mybert.pkl
TOTAL = 0.908 
0 = 0.91
1 = 0.907
2 = 0.906

Eval on total_df
TOTAL = 0.981
Nein = 0.982 on 3965 samples
Metapher = 0.973 on 735 samples
Kandidat = 0.981 on 3260 samples
"""