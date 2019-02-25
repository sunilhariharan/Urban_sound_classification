I have used the kaggle dataset(Urban sound classification) for the challenge given to me.

Here is the link to the dataset - https://www.kaggle.com/pavansanagapati/urban-sound-classification

1)My first step was to do some exploratory data analysis of the dataset.
2)I then started with mfcc feature extraction and encoding of the target variable.
3)I created a train and validation split out of the train dataset provided to me for a binary classification problem.
4)I built several models...started with random forest,went on to build sequential neural network models using Keras.I have built a base dense model,smaller(1 layered dense model),larger(2 layered dense model) and finally 1 layer GRU+dense model.
5)I finally chose the smaller dense model to be the the final model for testing data only because it was talking half the time to train and accuracy and auc were almost similar.
6)Accuracy_score of my final model is 0.9907 and auc_score is 0.9919
