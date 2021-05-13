from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy
import tensorflow.compat.v1 as tf

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('.sonar.csv', header = None)
#print(df.info())
#print(df.head())

dataset = df.values
X = dataset[, 060].astype(float) 
# 텐서플로우 2.0 부터 keras의 type이 더 strict하게 작용해 에러 발생
# 형을 입력해줌
Y_obj = dataset[, 60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state = seed)

model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=200, batch_size=5)
print(n Accuracy %.4f % (model.evaluate(X_test, Y_test)[1]))