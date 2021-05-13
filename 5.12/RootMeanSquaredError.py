import numpy as np

ab = [3, 76]
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

def predict(x): # 일차 방정식 y=ax+b를 구하는 함수
  return ab[0]*x + ab[1]

def rmse(p, a): # 평균 제곱근을 구하기 위한 함수
  return np.sqrt(((p-a) ** 2).mean())

def rmse_val(predict_result, y): # rmse() 함수에 데이터를 대입하여 최종값을 구하는 함수
  return rmse(np.array(predict_result), np.array(y))

predict_result = []

for i in range(len(x)):
  predict_result.append(predict(x[i]))
  print("공부 시간 = %.f, 실제 점수 = %.f, 예측 점수 = %.f" % (x[i], y[i], predict(x[i])))

print("rmse 최종값 : " + str(rmse_val(predict_result, y)))

# a = 3, b = 76인 경우 3.316...의 오차값을 가지게됨