from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy
import tensorflow.compat.v1 as tf

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('./sonar.csv', header = None)
#print(df.info())
#print(df.head())

dataset = df.values
X = dataset[:, 0:60].astype(float) 
# 텐서플로우 2.0 부터 keras의 type이 더 strict하게 작용해 에러 발생
# 형을 입력해줌
Y_obj = dataset[:, 60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=200, batch_size=5)
print("\n Accuracy: %.4f" % (model.evaluate(X,Y)[1]))
# 과적합(overfitting) 발생