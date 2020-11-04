import numpy as np
from random import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
import csv

#leaky relu seems to take away plateau of 87 with relu
def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)

def generate_dataset(test_size):
    ######################################## X DATA
    
    #input data with date,index1,homefield1,index2,homefield2

    #creates single array containing all game data (strings of each element)
    all_games = []
    x = []
    with open("./data/gameswobracket.txt", 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for arr in reader:
            for item in arr:
                all_games.append(item.strip())
    #fixes formatting from splitting on delimiter ',' since each game is also seperated by space
    i=0
    while i < len(all_games):
        word = all_games[i]
        if ' ' in word:
            words = word.split(" ")
            all_games[i] = words[0]
            all_games.insert(i+1, words[1])
        i+=1

    #puts each game into its own 'vector'
    count = 0
    game = []
    for element in all_games:
        if count < 8:
            game.append(int(element))
            count += 1
        else:
            #puts finalized vector into array
            x.append(game)
            count = 0
            game = []
            game.append(int(element))
            count += 1
    #removes date and scores
    for item in x:
        item.pop(1)
        item.pop(3)
        item.pop(5)
    
    x = np.array(x)
######################################## Y DATA
    games = []
    x2 = []
    y = []
    with open("./data/gameswobracket.txt", 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for arr in reader:
            for item in arr:
                games.append(item.strip())
    i=0
    while i < len(games):
        word = games[i]
        if ' ' in word:
            words = word.split(" ")
            games[i] = words[0]
            games.insert(i+1, words[1])
        i+=1

    count = 0
    game = []
    for element in games:
        if count < 8:
            game.append(int(element))
            count += 1
        else:
            x2.append(game)
            count = 0
            game = []
            game.append(int(element))
            count += 1
    
    #if team 1 beats team 2 then we add team 1's id, else we add team 2s id
    for item in x2:
        if item[4] > item[7]:
            tmp = []
            tmp.append(item[2])
            y.append(tmp)
        else:
            tmp = []
            tmp.append(item[5])
            y.append(tmp)
            
    y = np.array(y)

    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = test_size)
    
    return x_train, x_test, y_train, y_test

####################################MAIN################################################
x_train, x_test, y_train, y_test = generate_dataset(.2)
    
#build model - takes in 5d vectors
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(5, activation = my_leaky_relu, input_shape = (5,)))
#256 is fairly arbitrary atm... but works??
model.add(tf.keras.layers.Dense(256,activation = my_leaky_relu))
#1d vector is output
model.add(tf.keras.layers.Dense(1))
#compile model - Adam is a good all around optimizedr
Optimizer = tf.keras.optimizers.Adam()
#MAE gives us the magnitude of the average error (without direction)
model.compile(optimizer=Optimizer, loss="mae")
#train
test = model.fit(x_train, y_train, epochs=5000, batch_size = 200)
#evaluate
model.evaluate(x_test,y_test,verbose=1)

#save model for later
model.save("saved_model")