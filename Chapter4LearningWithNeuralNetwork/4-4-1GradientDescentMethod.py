# 勾配降下法
import numpy as np


def numerical_gradient(f, x):
  h = 1e-4 # 0.0001
  grad = np.zeros_like(x) # xと同じ形状の配列を生成

  for idx in range(x.size):
    tmp_val = x[idx]
    
    # f(x+h)の計算
    x[idx] = tmp_val + h
    fxh1 = f(x)

    # f(x-h)の計算
    x[idx] = tmp_val - h
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2*h)
    x[idx] = tmp_val # 値を元に戻す
  
  return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
  # init_xは初期値、lrはlearning rate、step_numは勾配法による繰り返しの数
  x = init_x
  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad
  
  return x


def function_2(x): # xはNumpy配列
  return x[0]**2 + x[1]**2


init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
# [-6.11110793e-10  8.14814391e-10]

# 学習率が大きすぎるケースと小さすぎるケースについて実験
# 大きすぎる例
print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))
# [ 2.34235971e+12 -3.96091057e+12]

# 小さすぎる例
print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))
# [ 2.34235971e+12 -3.96091057e+12]