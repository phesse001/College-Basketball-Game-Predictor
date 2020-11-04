#########################################Format input data
import tensorflow as tf
import csv
model = tf.keras.models.load_model('saved_model')

z = []
with open("./data/tournament.txt", 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for arr in reader:
        for item in arr:
            z.append(item.strip())
i=0
while i < len(z):
    word = z[i]
    if ' ' in word:
        words = word.split(" ")
        z[i] = words[0]
        z.insert(i+1, words[1])
    i+=1

count = 0
tmp = []
x3 = []
for element in z:
    if count < 8:
        tmp.append(int(element))
        count += 1
    else:
        x3.append(tmp)
        count = 0
        tmp = []
        tmp.append(int(element))
        count += 1
#tournament data formatted same as x data, now we can find the team id's from which teams actually won
actual = []
for game in x3:
    if(game[4] > game[7]):
        actual.append(game[2])
    else:
        actual.append(game[5])
        
for item in x3:
    item.pop(1)
    item.pop(3)
    item.pop(5)

#loop through tournament data, with each input having the 5 features(days,team1ID,hfa,team2ID,hfa)
predictions = []
tmp = []
for data in x3:
    tmp.append(data)
    prediction = model.predict(tmp)
    winner = None
    #find out which id prediction was closest to
    diff1 = abs(prediction - data[1])
    diff2 = abs(prediction - data[3])
    if(diff1 < diff2):
        winner = data[1]
    else:
        winner = data[3]
    #print(str(data[1]) + " VS " + str(data[3]) + " - Winner: " + str(winner))
    predictions.append(winner)
    tmp = []

num_wrong = 0
i = 0
size = len(actual)
while(i < size):
    if predictions[i] != actual[i]:
        num_wrong +=1
    print("Predicted: " + str(predictions[i]) + " Actual: " + str(actual[i]))
    i +=1
print("\nTotal number of games incorrectly predicted: " + str(num_wrong) + "\n")
#total correct/total
print("Percent correct: " + str(100*(size - num_wrong)/size))