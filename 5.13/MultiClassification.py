from keras.models import Sequential
from keras.layers.core import Dense
import pandas as pd
import numpy
import tensorflow.compat.v1 as tf

df = pd.read_csv('./iris.csv', names =["sepal_legnth", "sepal_width", "petal_length", "petal_width", "species"])
print(df.head())

import seaborn as sns
import matplotlib.pyplot as plt

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

#sns.pairplot(df, hue='species');
#plt.show()

dataset = df.values
X = dataset[:,0:4].astype(float)
Y_obj = dataset[:,4]

# Y는 숫자가 아닌 문자열임으로 원 핫 인코딩을 통해 숫자로 바꾸어줌
from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 활성화 함수 적용을 위해 Y의 값을 0, 1로 바꾸어줌
from keras.utils import np_utils
Y_encoded = np_utils.to_categorical(Y)

#print(Y_encoded)

model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.fit(X, Y_encoded, epochs=50, batch_size=1)

print("\n Accuracy: %.4f" % (model.evaluate(X, Y_encoded)[1]))

