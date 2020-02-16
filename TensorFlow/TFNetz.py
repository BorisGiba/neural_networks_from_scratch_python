#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[ ]:


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
anzahlEinträge=60000
x_train = mnist.train.images[:anzahlEinträge,:]
y_train = mnist.train.labels[:anzahlEinträge,:]

x_train=x_train[:10000]
y_train=y_train[:10000]

x_test = mnist.test.images[:anzahlEinträge,:]
y_test = mnist.test.labels[:anzahlEinträge,:]


# In[ ]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(526,tf.nn.relu))
model.add(tf.keras.layers.Dense(268,tf.nn.relu))
model.add(tf.keras.layers.Dense(10,tf.nn.softmax))

model.compile(optimizer="adam",
                     loss="categorical_crossentropy",
                     metrics=["accuracy"])


# In[ ]:


model.fit(x_train,y_train,epochs=1)


# In[ ]:


model.evaluate(x_test,y_test)

