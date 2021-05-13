from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from keras.models import load_model

import pandas as pd
import numpy
import tensorflow.compat.v1 as tf

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('sonar.csv', header = None)
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

n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

accuracy = []

for train, test in skf.split(X, Y):
  model = Sequential()
  model.add(Dense(24, input_dim=60, activation='relu'))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

  model.fit(X[train], Y[train], epochs=100, batch_size=5)
  k_accuracy = "%.4f" % (model.evaluate(X[test], Y[test])[1])
  accuracy.append(k_accuracy)

print("\n %.f fold accuracy: " % n_fold, accuracy)