import pandas as pd
df = pd.read_csv('./data/pima-indians-diabetes.csv', names = ["pregnant", 'plasma', 'pressure', 'thickness', 'insulin', 'BMI', 'pedigree', 'age', 'class'], header = 0 )

#print(df.head(5))
#print(df.info())
#print(df.describe())
#print(df[['pregnant', 'class']])

#print(df[['pregnant', 'class']].groupby(['pregnant'], as_index=False).mean().sort_values(by='pregnant', ascending=True))
# gropuby 함수를 사용해 pregnant 정보를 기준으로 하는 새 그룹을 생성
# as_index를 통해 새로운 index 생성
# mean() 함수를 사용해 class의 평균을 구하고, 이를 sort_value() 함수로 pregnant의 오름차순으로 정리

import matplotlib.pyplot as plt
import seaborn as sns

# 그래프 크기 결정
plt.figure(figsize=(12, 12))

# heatmap 함수는 각 항목간의 상관관계를 나타내줌
# 두 항목이 전혀 다른 패턴으로 변화하면 0을
# 서로 비슷한 패턴으로 변화할수록 1에 가까운 값을 출력
sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor='white', annot=True)

plt.show()

# plasma와 class 항목만 따로 출력
grid = sns.FacetGrid(df, col = 'class')
grid.map(plt.hist, 'plasma', bins=10)
plt.show()

from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow.compat.v1 as tf

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

dataset = numpy.loadtxt('./data/pima-indians-diabetes.csv', skiprows=1, delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=200, batch_size=10)

print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))