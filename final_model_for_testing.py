
# coding: utf-8

# # final model for test dataset and deployment (smaller model) because training time is half than the larger model and also the results are almost same

# In[ ]:


def final_model(X_test):
    pred = pipeline1.predict(X_test)
    print(pred)

