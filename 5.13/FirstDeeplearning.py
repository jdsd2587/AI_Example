from keras.models import Sequential
from keras.layers import Dense

import numpy
import tensorflow.compat.v1 as tf

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

Data_set = numpy.loadtxt("./data/ThoraricSurgery.csv", delimiter=',')

X = Data_set[:,0:17]
Y = Data_set[:,17]

# 딥러닝의 구조를 짜고 층을 설정하는 코드
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu')) # 은닉층
# 이 층에 30개의 노드를 만들고, 입력 데이터로 몇 개의 값이 들어올지 설정(17)
# keras는 입력층을 따로 만드는 것이 아니라 input_dim을 통해 첫 번째 층이 은닉층 + 입력층 역할
# 여기서는 17개의 값을 받아 30개의 노드로 보내겠다는 뜻, 활성화 함수 ReLU
model.add(Dense(1, activation = 'sigmoid')) # 출력층
# 출력층이므로 노드 수 1개, 활성화 함수 Sigmoid

# 지정한 모델을 컴파일 하는 코드
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
# 오차는 평균 제곱 오차를 사용
# 최적화(가중치 수정)는 adam을 사용 -> adam 이 현재 가장 많이 사용되는 고급 경사 하강법
# metrics 함수는 모델이 컴파일될 때 모델 수행 결과를 나타내게끔 설정하는 부분
#  -> 정확도를 측정하기 위해 사용되는 테스트 샘플을 학습 과정에서 제외시킴으로써 over fitting 문제를 방지

model.fit(X, Y, epochs=30, batch_size=10)
# 학습이 모든 샘플에 대해 1회 실행되는 것을 1 epoch
#  -> 여기서는 30회 실행하라는 뜻
# batch_size는 샘플을 한번에 몇 개씩 끊어서 처리할지 설정
#  -> 여기서는 470개의 샘플을 10개씩 끊어서 넣게됨

print("\n Accuracy : %.4f" % (model.evaluate(X, Y)[1]))