README

To run see function main in main.py

Details to run:

Parameters:

CONFIG['model'] : the model to run
Values : 'Conv' or 'BERT'

CONFIG['Tasks'] : Tasks to consider
Values : ['genre','violence','romantic','sadness',
,'danceability']

Always include 'genre' 

Exmaples:
 {'Model':'Conv','Tasks':['genre']}
 {'Model':'BERT','Tasks':['genre','violence','romantic']}
 {'Model':'Conv','Tasks':['genre','romantic']}


NUM_EX - number of experiments to run 
Set at default 5


    """