#!/usr/bin/env python
# coding: utf-8

# # Assignment 1

# ### Title : Develop an MNIST hand written digit recognition system using Artificial Neural Network and by Convolutional Neural network.
# 
# ### Description : MNIST Dataset is a large dataset of handwritten digits containing 60,000 images. The candidate is supposed to develop ANN based and CNN based system for the digit recognition classifer.
# 
# ### Objective: Familiarity with creating suitable architecture of CNN and compare the classifier results with ANN to prove its validity.

# ##### Importing Libraries

# In[1]:


import numpy as np


# In[2]:


import tensorflow as tf


# In[3]:


from tensorflow.keras import datasets,layers,models


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


from sklearn.metrics import confusion_matrix,classification_report


# ##### Loading Dataset(MNIST)

# In[6]:


(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()


# In[7]:


X_train.shape


# ##### We have total 60,000 images while size of the image being 28 * 28

# In[8]:


X_train[0]


# In[9]:


plt.figure(figsize=(25,5))


# In[10]:


plt.imshow(X_train[0])


# ##### Here we define a list for labelling the numbers in the dataset from Zero to Nine

# In[11]:


classes=["zero","one","two","three","four","five","six","seven","eight","nine"]


# In[12]:


y_train.shape


# In[13]:


y_train


# ##### Function for plotting the image with the label

# In[14]:


def plt_show(X,y,index):
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[15]:


y_train = y_train.reshape(-1,)


# In[16]:


y_train


# In[17]:


X_train[0] = X_train[0]/255


# In[18]:


X_train[0]


# In[19]:


X_train = X_train/255


# In[20]:


plt_show(X_train,y_train,45)


# ##### Now we will start creating a ANN model,starting with defining the model.
# ##### Here we have used 3 hidden layer, a flatten layer for making it linear and an output layer.

# In[21]:


ann=models.Sequential([layers.Flatten(input_shape=(28,28)),
                       layers.Dense(5000,activation='relu'),
                       layers.Dense(500,activation='relu'),
                       layers.Dense(100,activation='relu'),
                       layers.Dense(10,activation='sigmoid')])


# ##### We will compile the model while using SGD as optimizer, sparse categorical crossentropy as loss and accuracy as metrices

# In[22]:


ann.compile(optimizer='SGD',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# ##### The model is compiled and we will fit the data in the model we just created, 
# ##### will keep backpropagating until we get a good accuracy.

# In[23]:


ann.fit(X_train,y_train,epochs=6)


# In[24]:


ann.evaluate(X_test,y_test)


# ##### The overall accuracy we get is 97%

# In[25]:


y_predict = ann.predict(X_test)


# In[26]:


y_predict


# In[27]:


y_predict_classes=[np.argmax(element) for element in y_predict]


# In[28]:


y_test = y_test.reshape(-1,)


# In[29]:


print("classification report\n",classification_report(y_test,y_predict_classes))


# In[ ]:


y_predict_classes


# In[31]:


X_train = np.expand_dims(X_train, axis=-1)


# ##### CNN model creation, we start with defining the model, creating two Conv2d layer since images are in 2d
# ##### Next we use MaxPooling here, we also add flatten layer so to convert it into a linear form
# ##### After this we add fully connected layer.

# In[41]:


cnn=models.Sequential([
                       layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),
                       layers.MaxPooling2D((2,2)),
                       layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
                       layers.MaxPooling2D((2,2)),                    
                       layers.Flatten(),
                       layers.Dense(64,activation='relu'),
                       layers.Dense(10,activation='softmax')])


# ##### We will compile the model while using SGD as optimizer, sparse categorical crossentropy as loss and accuracy as metrices

# In[42]:


cnn.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# ##### The model is compiled and we will fit the data in the model we just created, 
# ##### will keep backpropagating until we get a good accuracy. We have used 5 epochs

# In[45]:


cnn.fit(X_train,y_train,epochs=1)


# In[47]:


X_test = np.expand_dims(X_test, axis=-1)


# In[48]:


cnn.evaluate(X_test,y_test)


# ##### The Accuracy we achive with cnn model is 98.83%

# In[49]:


y_test=y_test.reshape(-1,)


# In[50]:


y_test


# In[51]:


plt_show(X_test,y_test,5)


# In[52]:


X_test = np.expand_dims(X_test, axis=-1)


# In[53]:


y_predict=cnn.predict(X_test)


# In[54]:


y_predict


# In[57]:


y_classes=[np.argmax(element) for element in y_predict]


# In[ ]:


y_classes


# ### Comparing ANN and CNN model the accuracy achieved after backpropagation with ann is 97% and with CNN it goes upto 99%.
