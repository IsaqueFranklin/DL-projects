what_were_covering = {
    1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model",
    6: "putting it all together"
}

import torch
from torch import nn #nn contains Pytorch's building blocks for neural networks
import matplotlib.pyplot as plt

#Check pytorch version
torch.__version__

# Basicamente eu crie um artificialmente conjunto de coisas X e as "labels"/classificações para as coisas X, chamadas Y
# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight * X * bias

X[:10], Y[:10]

#Vamos dividir os dados entre dados de treinamento e de sets de teste

#The testing set is basically the labeled data, the batch in which there are relations between things X and labels Y.
#The validation set is where the model gets tunned.
#The training set is the data where the model learns from.

## Aqui vamos usar apenas o training e testing sets

## Note: When dealing with real-world data, 
##this step is typically done right at the start of a project (the test set should always be kept separate from all other data). 
##We want our model to learn on training data and then evaluate it on test data to get an indication of how well it generalizes to unseen examples.

# Create train/test split
train_split = int(0.8 * len(X)) #80% of data for training set, 20% for testing
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]

len(X_train), len(Y_train), len(X_test), len(Y_test)

#Let's create a function to visualize our data

def plot_predictions(train_data=X_train,
                     train_labels=Y_train,
                     test_data=X_test,
                     test_labels=Y_test,
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  #Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

  #Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testint data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  #Show the legend
  plt.legend(prop={"size": 14})

plot_predictions();