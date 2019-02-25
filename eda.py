
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


# In[4]:


df['Class'].value_counts()


# In[5]:


df['Class'].value_counts().plot.bar()


# # jackhammer

# In[6]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/33.wav')


# In[7]:


data1, sampling_rate1 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/33.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data1, sr=sampling_rate1)
plt.title('jackhammer')


# # engine_idling

# In[8]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/17.wav')


# In[9]:


data2, sampling_rate2 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/17.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data2, sr=sampling_rate2)
plt.title('engine_idling')


# # siren

# In[10]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/0.wav')


# In[11]:


data3, sampling_rate3 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/0.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data3, sr=sampling_rate3)
plt.title('siren')


# # air_conditioner

# In[12]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/22.wav')


# In[13]:


data4, sampling_rate4 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/22.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data4, sr=sampling_rate4)
plt.title('air_conditioner')


# # dog_bark

# In[14]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/4.wav')


# In[15]:


data5, sampling_rate5 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/4.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data5, sr=sampling_rate5)
plt.title('dog_bark')


# # children_playing

# In[16]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/6.wav')


# In[17]:


data6, sampling_rate6 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/6.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data6, sr=sampling_rate6)
plt.title('children_playing')


# # street_music

# In[18]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/1.wav')


# In[19]:


data7, sampling_rate7 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/1.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data7, sr=sampling_rate7)
plt.title('street_music')


# # drilling

# In[20]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/2.wav')


# In[21]:


data8, sampling_rate8 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/2.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data8, sr=sampling_rate8)
plt.title('drilling')


# # car_horn

# In[22]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/8693.wav')


# In[23]:


data9, sampling_rate9 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/8693.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data9, sr=sampling_rate9)
plt.title('car_horn')


# # gun_shot

# In[24]:


import IPython.display as ipd
ipd.Audio('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/12.wav')


# In[25]:


data10, sampling_rate10 = librosa.load('/Users/sunilhariharan/Downloads/data science/datasets/hiring challenge/train/Train/12.wav')

import librosa.display
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data10, sr=sampling_rate10)
plt.title('gun_shot')

