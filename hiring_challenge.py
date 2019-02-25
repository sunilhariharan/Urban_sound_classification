
# coding: utf-8

# In[1]:


import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # loading the train dataset (urban sound classification--kaggle) 

# In[2]:


df=pd.read_csv('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/train.csv')


# # exploratory data analysis

# In[3]:


df


# In[6]:


df['Class'].value_counts()


# In[7]:


df['Class'].value_counts().plot.bar()


# # jackhammer

# In[ ]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/33.wav')


# In[ ]:


data1, sampling_rate1 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/33.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data1, sr=sampling_rate1)
plt.title('jackhammer')


# # engine_idling

# In[ ]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/17.wav')


# In[ ]:


data2, sampling_rate2 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/17.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data2, sr=sampling_rate2)
plt.title('engine_idling')


# # siren

# In[ ]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/0.wav')


# In[ ]:


data3, sampling_rate3 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/0.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data3, sr=sampling_rate3)
plt.title('siren')


# # air_conditioner

# In[ ]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/22.wav')


# In[ ]:


data4, sampling_rate4 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/22.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data4, sr=sampling_rate4)
plt.title('air_conditioner')


# # dog_bark

# In[ ]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/4.wav')


# In[ ]:


data5, sampling_rate5 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/4.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data5, sr=sampling_rate5)
plt.title('dog_bark')


# # children_playing

# In[ ]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/6.wav')


# In[ ]:


data6, sampling_rate6 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/6.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data6, sr=sampling_rate6)
plt.title('children_playing')


# # street_music

# In[ ]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/1.wav')


# In[ ]:


data7, sampling_rate7 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/1.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data7, sr=sampling_rate7)
plt.title('street_music')


# # drilling

# In[ ]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/2.wav')


# In[ ]:


data8, sampling_rate8 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/2.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data8, sr=sampling_rate8)
plt.title('drilling')


# # car_horn

# In[ ]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/8693.wav')


# In[ ]:


data9, sampling_rate9 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/8693.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data9, sr=sampling_rate9)
plt.title('car_horn')


# # gun_shot

# In[ ]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/12.wav')


# In[ ]:


data10, sampling_rate10 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/12.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data10, sr=sampling_rate10)
plt.title('gun_shot')


# # python code to extract MFCC feature

# In[12]:


path='/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/'
def mfcc_extraction(dataframe,ID):
    l=dataframe[ID].values.tolist()
    appended=[]
    for i in l:
        data, sampling_rate = librosa.load(path+str(i)+'.wav')
        mfccs=np.mean(librosa.feature.mfcc(y=data,sr=sampling_rate,n_mfcc=20).T,axis=0)
        appended.append(mfccs)
    df_mfcc=pd.DataFrame(appended,columns=list(range(20)))
    return df_mfcc
        


# # train and validation dataset for two voice keywords(binary classification) ---- selected gun_shot and car_horn(2 keywords with less counts) keeping in mind the computational power required

# In[ ]:


df1=df[df['Class']=='gun_shot']


# In[ ]:


df2=df[df['Class']=='car_horn']


# In[ ]:


df=pd.concat([df1,df2])


# In[ ]:


df


# In[13]:


df_train=mfcc_extraction(df,'ID')


# In[14]:


df_train


# In[15]:


def encoding(dataframe,target): 
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    dataframe[target] = le.fit_transform(dataframe[target])
    dataframe.reset_index(inplace=True)
    return dataframe[target]


# In[16]:


df_train['Class']=encoding(df,'Class')


# In[17]:


df_train


# In[73]:


def split(dataframe):
    from sklearn.model_selection import train_test_split
    X = df_train.iloc[:,0:20]
    y = df_train.iloc[:,20]
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2,random_state=9)
    return X_train,X_val,y_train,y_val,X,y


# In[74]:


X_train,X_val,y_train,y_val,X,y=split(df_train)


# # model training using Random Forest

# In[134]:


def base_model(X_train,X_val,y_train,y_val,X,y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import cross_val_score
    clf_1 = RandomForestClassifier()
    clf_1.fit(X_train,y_train)
    y_clf_1 = clf_1.predict(X_val)
    accuracy_clf_1 = accuracy_score(y_val, y_clf_1)
    auc_roc1 = roc_auc_score(y_val, y_clf_1) 
    print("Accuracy score of the model is: {}".format(accuracy_clf_1))
    print("roc_auc_score of the model is: {}".format(auc_roc1))
    print(cross_val_score(clf_1, X, y, cv=5))


# In[138]:


model(X_train,X_val,y_train,y_val,X,y)


# In[143]:


def tuned_model(X_train,X_val,y_train,y_val,X,y):
    params = {'n_estimators': [10,100,1000], 
               
              'criterion': ['entropy', 'gini'], 
              
             }
    from sklearn.model_selection import RandomizedSearchCV,cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import cross_val_score
    clf_2 = RandomizedSearchCV(RandomForestClassifier(),param_distributions = params,n_iter=5)
    clf_2.fit(X_train,y_train)
    y_clf_2 = clf_2.predict(X_val)
    accuracy_clf_2 = accuracy_score(y_val, y_clf_2)
    auc_roc2 = roc_auc_score(y_val, y_clf_2) 
    print("Accuracy score of the model is: {}".format(accuracy_clf_2))
    print("roc_auc_score of the model is: {}".format(auc_roc2))
    print(cross_val_score(clf_2, X, y, cv=5))
    return clf_2.best_estimator_


# In[144]:


tuned_model(X_train,X_val,y_train,y_val,X,y)


# # binary classification using Keras deep learning library

# In[111]:


import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# In[112]:


#baseline model
def create_baseline():
    model = Sequential()
    model.add(Dense(20, input_dim=20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

seed = 7
numpy.random.seed(seed)
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

estimator.fit(X_train, y_train)
prediction = estimator.predict(X_val)
accuracy_score(y_val, prediction)


# In[114]:


#evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

pipeline.fit(X_train, y_train)
prediction1 = pipeline.predict(X_val)
accuracy_score(y_val, prediction1)


# # tuning layers and number of neurons in the model

# In[116]:


#smaller model
def create_smaller():
    model = Sequential()
    model.add(Dense(10, input_dim=20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=5, verbose=0)))
pipeline1 = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

pipeline1.fit(X_train, y_train)
prediction_a = pipeline1.predict(X_val)
accuracy_score(y_val, prediction_a)


# In[122]:


from sklearn.metrics import roc_curve
prediction_a = pipeline1.predict(X_val).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_val, prediction_a)


# In[126]:


from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)


# In[127]:


auc_keras


# In[128]:


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[117]:


#larger model
def create_larger():
    model = Sequential()
    model.add(Dense(20, input_dim=20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=100, batch_size=5, verbose=0)))
pipeline2 = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

pipeline2.fit(X_train, y_train)
prediction_b = pipeline2.predict(X_val)
accuracy_score(y_val, prediction_b)


# # 1 Layers GRU+Dense

# In[3]:


from keras.layers import GRU, Dense
from keras.models import Sequential
from keras.regularizers import l2


# In[ ]:


class DeepNetArch2L1: 
    def __init__(self, sl, initial_lr, l2_reg, dropout, rec_dropout, optimizer, summary):
        self.sl = sl
        self.summary = summary
        self.l2_reg = l2(l2_reg)
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.initial_lr = initial_lr
        self.optimizer = optimizer

    def arch_generator(self):
        model_name = "arch2l1"
        model = Sequential()
        model.add(GRU(units=self.sl, return_sequences=False, dropout=self.dropout, recurrent_dropout=self.rec_dropout,
                      input_shape=(self.sl, 1), stateful=False))
        model.add(Dense(1, activation="sigmoid", kernel_initializer="he_normal", kernel_regularizer=self.l2_reg))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        if self.summary:
            print(model.summary())
        return model, model_name


# # final model for test dataset and deployment (smaller model) because training time is half than the larger model and also the results are almost same

# In[120]:


def final_model(X_test):
    pred = pipeline1.predict(X_test)
    print(pred)

