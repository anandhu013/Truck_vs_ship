import tensorflow as tf
import numpy as np
import joblib
from keras.datasets import cifar10
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from keras.layers import Input

(trainX,trainY),(testX,testY)=cifar10.load_data()
trX=[]
trY=[]
teX=[]
teY=[]
for ind,pic in enumerate(trainY):
  if(pic[0]==8):
    trX.append(trainX[ind])
    trY.append([0])
  if(pic[0]==9):
    trX.append(trainX[ind])
    trY.append([1])
for ind,pic in enumerate(testY):
  if(pic[0]==8):
    teX.append(testX[ind])
    teY.append([0])
  if(pic[0]==9):
    teX.append(testX[ind])
    teY.append([1])

trY=np.array(trY)
trX=np.array(trX)
teX=np.array(teX)
teY=np.array(teY)
trX = trX.astype('float32')
teX = teX.astype('float32')
trX  /= 255.0
teX /= 255.0

def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model


model=define_model()
history=model.fit(trX,trY,batch_size=32,epochs=50,validation_data=(teX,teY))

model.save("model.h5")