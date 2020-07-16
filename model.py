#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


high_dir = 'datasets/high/'
low_dir = 'datasets/low/'



def generate_3darray(directory):
    files = os.listdir(directory)
    dims = np.genfromtxt(directory + files[0], delimiter=',').shape
    length = len(files)
    arr = np.zeros((length, dims[0], dims[1]))
    
    for idx, val in enumerate(files):
        arr[idx] = np.genfromtxt(directory + val, delimiter=',')
        
    return arr

hi = generate_3darray(high_dir)
lo = generate_3darray(low_dir)

X = np.concatenate((hi, lo))

hi_Y = np.ones(len(hi), dtype=int)
lo_Y = np.zeros(len(lo), dtype=int)

Y = np.concatenate((hi_Y, lo_Y))


# In[3]:


train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1)


# In[4]:


print('Training data shape: ', train_X.shape, train_Y.shape)
print('Testing data shape: ', test_X.shape, test_Y.shape)


# In[5]:


classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs: ', nClasses)
print('Output classes: ', classes)


# In[6]:


plt.figure(figsize=[10,2])

# Display first image in training set
plt.subplot(121)
plt.pcolormesh(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth: {}".format(train_Y[0]))

# Display first image in testing set
plt.subplot(122)
plt.pcolormesh(test_X[0], cmap='gray')
plt.title("Ground Truth: {}".format(test_Y[0]))


# In[7]:


ydim = train_X[0].shape[0]
xdim = train_X[0].shape[1]
train_X = train_X.reshape(-1, ydim, xdim, 1)
test_X = test_X.reshape(-1, ydim, xdim, 1)
train_X.shape, test_X.shape


# In[8]:


train_X = train_X.astype('float32')
test_X = test_X.astype('float32')


# In[9]:


# Change labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# See the difference?
print('Original label: ', train_Y[0])
print('After conversion to one-hot: ', train_Y_one_hot[0])


# In[10]:


train_Y_one_hot.shape


# In[11]:


# Partition data for 80% training and 20% validation
# should reduce overfitting and boost test performance

train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2)

train_X.shape, valid_X.shape, train_label.shape, valid_label.shape


# In[12]:


# let's get down to business
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


# In[13]:


# Parameters
batch_size = 64
epochs = 50
num_classes = nClasses


# In[14]:


emg_model = Sequential()
emg_model.add(
    Conv2D(
        32, 
        kernel_size=(3, 3), 
        activation='linear', 
        input_shape=(ydim, xdim, 1), 
        padding='same'))
emg_model.add(LeakyReLU(alpha=0.1))
emg_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
emg_model.add(Dropout(0.25))
emg_model.add(
    Conv2D(
        64,
        (3, 3),
        activation='linear',
        padding='same'))
emg_model.add(LeakyReLU(alpha=0.1))
emg_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
emg_model.add(Dropout(0.25))
emg_model.add(
    Conv2D(
        128,
        (3, 3),
        activation='linear',
        padding='same'))
emg_model.add(LeakyReLU(alpha=0.1))
emg_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
emg_model.add(Dropout(0.4))
emg_model.add(Flatten())
emg_model.add(Dense(128, activation='linear'))
emg_model.add(LeakyReLU(alpha=0.1))
emg_model.add(Dropout(0.3))
emg_model.add(Dense(num_classes, activation='softmax'))


# In[15]:


emg_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'])


# In[16]:


emg_model.summary()


# In[17]:


emg_train = emg_model.fit(
    train_X,
    train_label,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valid_X, valid_label))


# In[18]:


test_eval = emg_model.evaluate(test_X, test_Y_one_hot, verbose=0)


# In[19]:


print('Test loss: ', test_eval[0])
print('Test accuracy: ', test_eval[1])


# In[20]:


# Visualization
accuracy = emg_train.history['accuracy']
val_accuracy = emg_train.history['val_accuracy']
loss = emg_train.history['loss']
val_loss = emg_train.history['val_loss']
epoch = range(len(accuracy))
plt.plot(epoch, accuracy, 'bo', label='Training accuracy')
plt.plot(epoch, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epoch, loss, 'bo', label='Training loss')
plt.plot(epoch, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[21]:


emg_train_dropout = emg_model.fit(
    train_X,
    train_label,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valid_X, valid_label))


# In[22]:


emg_model.save("emg_model_dropout.h5py")


# In[23]:


test_eval= emg_model.evaluate(test_X, test_Y_one_hot, verbose=1)


# In[24]:


print('Test loss: ', test_eval[0])
print('Test accuracy: ', test_eval[1])


# In[25]:


# Visualization
accuracy = emg_train_dropout.history['accuracy']
val_accuracy = emg_train_dropout.history['val_accuracy']
loss = emg_train_dropout.history['loss']
val_loss = emg_train_dropout.history['val_loss']
epoch = range(len(accuracy))
plt.plot(epoch, accuracy, 'bo', label='Training accuracy')
plt.plot(epoch, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epoch, loss, 'bo', label='Training loss')
plt.plot(epoch, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[26]:


predicted_classes = emg_model.predict(test_X)


# In[27]:


predicted_classes = np.argmax(np.round(predicted_classes), axis=1)


# In[28]:


predicted_classes.shape, test_Y.shape


# In[29]:


correct = np.where(predicted_classes==test_Y)[0]
print("Found %d correct labels" % len(correct))
for i, correct, in enumerate(correct[:9]):
    plt.subplot(3, 3, i+1)
    plt.pcolormesh(test_X[correct].reshape(ydim, xdim), cmap='jet')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()


# In[30]:


incorrect = np.where(predicted_classes!=test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect, in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i+1)
    plt.pcolormesh(test_X[incorrect].reshape(ydim, xdim), cmap='jet')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()


# In[31]:


from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))


# In[ ]:




