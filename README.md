# NN-ML-Resources

## How To Run Model

Dependencies: python3, numpy, sklearn, and tensorflow

Inside of a terminal, simply navigate to this directory where you'll find 'main.py'. If you want to use the existing model, run 'python main.py'. 

It will print out the ID's of teams that won the particular game and next to that print the ID of the team the model predicted to win. 

If you would like to train the model yourself you can tune different model parameters in the 'train.py' script and run 'python train.py' to train the model yourself. It will save the last trained model to be used in 'main.py'

### Jupyter Notebooks

If you have jupyter notebooks installed you can also call 'jupyter notebook' in the terminal and navigate to the corresponding notebook 'MM-NN.ipynb'.

## General Information I've found while researching

Clearing up confusion about the purpose of activation functions - [Link](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)

Explanation of different size parameters in TF - [Link](https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc)

Rules of thumb regarding # of neurons in hidden layer:

"The number of hidden neurons should be between the size of the input layer and the size of the output layer.
The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
The number of hidden neurons should be less than twice the size of the input layer." - [Link](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)

Epochs and batch sizes - [Link](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9#:~:text=the%20data%20given.-,Epochs,it%20in%20several%20smaller%20batches.)

