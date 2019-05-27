# 手書き数字を学習するニューラルネットワークを実装
# 2層のニューラルネットワーク(隠れ層が1層)
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    # input_sizeは入力層のニューロン数、hidden_sizeは隠れ層のそれ、output_sizeは出力層のそれ、
    # 重みの初期化
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
  
  def predict(self, x):
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = sigmoid(a2)

    return y

  # x:入力データ、t:教師データ
  def loss(self, x, t):
    y = self.predict(x)

    return cross_entropy_error(y, t)
  
  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

  # x:入力データ、t:教師データ
  def numerical_gradient(self, x, t):
    loss_W = lambda W: self.loss(x, t)

    grads = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    return grads


net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape) # (784, 100)
print(net.params['b1'].shape) # (100,)
print(net.params['W2'].shape) # (100, 10)
print(net.params['b2'].shape) # (10,)


x = np.random.rand(100, 784) # ダミーの入力データ(100枚分)
t = np.random.rand(100, 10) # ダミーの正解ラベル(100枚分)

y = net.predict(x)

grads = net.numerical_gradient(x, t) # 勾配の計算
print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)

# numerical_gradient(数値微分による各パラメータの損失関数の勾配の計算)だと学習が遅く、gradsの出力までに時間がかかりすぎる
# 次の章で誤差逆伝播法(gradient)を導入し、学習時間を高速にする
