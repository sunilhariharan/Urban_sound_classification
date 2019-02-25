
# coding: utf-8

# # train and validation dataset for two voice keywords(binary classification) ---- selected gun_shot and car_horn(2 keywords with less counts) keeping in mind the computational power required

# In[ ]:


df1=df[df['Class']=='gun_shot']


# In[ ]:


df2=df[df['Class']=='car_horn']


# In[ ]:


df=pd.concat([df1,df2])


# In[ ]:


df


# In[ ]:


df_train=mfcc_extraction(df,'ID')


# In[ ]:


df_train


# In[ ]:


def encoding(dataframe,target): 
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    dataframe[target] = le.fit_transform(dataframe[target])
    dataframe.reset_index(inplace=True)
    return dataframe[target]


# In[ ]:


df_train['Class']=encoding(df,'Class')


# In[ ]:


df_train


# In[ ]:


def split(dataframe):
    from sklearn.model_selection import train_test_split
    X = df_train.iloc[:,0:20]
    y = df_train.iloc[:,20]
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2,random_state=9)
    return X_train,X_val,y_train,y_val,X,y


# In[ ]:


X_train,X_val,y_train,y_val,X,y=split(df_train)

