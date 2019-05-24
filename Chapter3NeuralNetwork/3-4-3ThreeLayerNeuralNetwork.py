# 3層ニューラルネットワークの実装(入力層、第一層、第二層、出力層)
# 活性化関数はシグモイド関数
# 出力層の活性化関数だけ恒等関数(入力をそのまま出力する)

import numpy as np

# 活性化関数の用意
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def identify_function(x):
  return x # 恒等関数

# 重みとバイアスの初期化
def init_network():
  network = {}
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network['b2'] = np.array([0.1, 0.2])
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])

  return network

def forward(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = identify_function(a3)

  return y

# 結果の検証
network = init_network()
x = np.array([0.1, 0.5])
y = forward(network, x)
print(y)
# [0.31234736 0.6863161 ]