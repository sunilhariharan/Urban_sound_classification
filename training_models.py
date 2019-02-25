
# coding: utf-8

# # model training using Random Forest

# In[ ]:


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


# In[ ]:


model(X_train,X_val,y_train,y_val,X,y)


# In[ ]:


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


# In[ ]:


tuned_model(X_train,X_val,y_train,y_val,X,y)


# # binary classification using Keras deep learning library

# In[ ]:


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


# In[ ]:


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


# In[ ]:


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

# In[ ]:


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


# In[ ]:


from sklearn.metrics import roc_curve
prediction_a = pipeline1.predict(X_val).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_val, prediction_a)


# In[ ]:


from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)


# In[ ]:


auc_keras


# In[ ]:


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[ ]:


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

# In[1]:


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

