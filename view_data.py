import pandas as pd

df1= pd.read_csv(r"data/metapher_gs.csv")
df1=df1[['Textstelle','Metapher?']]

df2= pd.read_csv(r"data/no_metapher1.csv")
df2=df2[['Textstelle','Metapher?']]



df= pd.concat([df1,df2])
df.rename(columns={"Textstelle":"text","Metapher?":"label"}, inplace=True)


print(df['label'].value_counts())

df['label'] = df['label'].replace(['Nein','Metapher','Metaphernkandidat'],[0,1,2])
print(df['label'].value_counts())

df.to_csv(r"data/original_data22.csv")