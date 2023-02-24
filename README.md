# DL4NLP
Project on German metaphors detection with BERT  

Link for the trained model: https://drive.google.com/drive/folders/1LCMFoLyAjPvjaCdQo8CB58-Pg28TjM38?usp=sharing  
Here you can find the two models, the "mybert.pkl" is a bert-base-german-cased model trained with 5 epochs and "historicalbert.pkl" is trained only with 2 epochs because we want to compare our results with another experiment.  

Here you can find two models, both based on "bert-base-german-cased". You can use GBert8 to test our model and produce our results or you can use GBert7 as BERT head and build an optimal classifier after the BERT head. 

The results are in: https://docs.google.com/spreadsheets/d/14h7gEPpEyMkUw5Hj9PwmtsP17HmCgHZ43sjn4vZJIv4/edit?usp=sharing

The filenames are intuitive. In the bottom of the train file you can also find the testing procedure, with and without cross evaluation. 
