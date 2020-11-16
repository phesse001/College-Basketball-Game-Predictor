#########################################Format input data
import tensorflow as tf
import csv
import os
import ck.kernel as ck

#load model
model = tf.keras.models.load_model('../saved_model')
#find path to dataset
i = {'action':'find','module_uoa':'dataset', 
     'data_uoa':'2019_teams_games'}

r = ck.access(i)
if r['return'] > 0:
    ck.err(r)
d_path = r['path']

#load raw data into array
r_data = []
with open(d_path + '/tournament.txt', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for arr in reader:
        for item in arr:
            r_data.append(item.strip())

#fix formatting issues with data(each vector is split with a space but individual items are split with commas)
i=0
while i < len(r_data):
    word = r_data[i]
    if ' ' in word:
        words = word.split(" ")
        r_data[i] = words[0]
        r_data.insert(i+1, words[1])
    i+=1

#create game vectors and store them in games array
count = 0
vect = []
games = []
for element in r_data:
    if count < 8:
        vect.append(int(element))
        count += 1
    else:
        games.append(vect)
        count = 0
        vect = []
        vect.append(int(element))
        count += 1

#tournament data formatted same as x data, now we can find the team id's from which teams actually won
#create an vector of correct results(actual) to compare to
actual = []
for game in games:
    if(game[4] > game[7]):
        actual.append(game[2])
    else:
        actual.append(game[5])
        
#removes date(YMD) and scores
for item in games:
    item.pop(1)
    item.pop(3)
    item.pop(5)

#loop through tournament data, make prediction for each input(game), with each input having the 5 features(days,team1ID,hfa,team2ID,hfa)
predictions = []
tmp = []
for game in games:
    tmp.append(game)
    prediction = model.predict(tmp)
    winner = None
    #find out which id the prediction was closest to
    diff1 = abs(prediction - game[1])
    diff2 = abs(prediction - game[3])
    if(diff1 < diff2):
        winner = game[1]
    else:
        winner = game[3]
    predictions.append(winner)
    tmp = []

#extract team names from file
ids_names = []
with open(d_path + '/teams.txt', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        ids_names.append(row)

num_wrong = 0
i = 0
size = len(actual)
pname = None
aname = None

print("--------------------------------\n| Predicted      "
	  "Actual        |\n--------------------------------")

while(i < size):
	#find associated name to id
    for item in ids_names:
	    if int(item[0]) == predictions[i]:
		    pname = item[1]
	    if int(item[0]) == actual[i]:
		    aname = item[1]
    if predictions[i] != actual[i]:
        num_wrong += 1
    #offset formatting
    os1 = 15 - len(pname)
    os2 = 15 - len(aname)
    print("|" + str(pname) + "".rjust(os1) + str(aname) + "".rjust(os2) + "|")
    i +=1
print("--------------------------------")
print("\nTotal number of games incorrectly predicted: " + str(num_wrong) + "\n")
print("Percent correct: " + str(100*(size - num_wrong)/size))