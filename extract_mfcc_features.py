
# coding: utf-8

# # python code to extract MFCC feature

# In[1]:


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

