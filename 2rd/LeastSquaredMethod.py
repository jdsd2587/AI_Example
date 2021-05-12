import numpy as np

x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

mx = np.mean(x) # x의 평균
my = np.mean(y) # y의 평균

divisor = sum([(mx-i)**2 for i in x]) # 최소제곱법의 분모

def top(x, mx, y, my):
  d = 0
  for i in range(len(x)):
    d += (x[i] - mx)  * (y[i] - my)
  return d
dividend = top(x, mx, y, my) # 최소제곱법의 분자

a = dividend / divisor # y = ax + b
b = my - (mx*a)
print("기울기 a = ", a)
print("y 절편 b = ", b)