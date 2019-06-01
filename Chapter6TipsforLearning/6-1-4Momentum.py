# SGD(stochastic gradient descent)(確率的勾配降下法)の欠点は、勾配の方向が本来の最小値ではない方向を指していることより、
# 関数の形状が投稿的でないと非効率な経路で探索することになる点。
# そこで、単に勾配方向へ進むよりももっとスマートな方法が求められる。
# Momentumの登場


import numpy as np

class Momentum:
  def __init__(self, lr=0.01, momentum=0.9):
    self.lr = lr
    self.momentum = momentum
    self.v = None
        
  def update(self, params, grads):
    if self.v is None:
      self.v = {}
      for key, val in params.items():
        self.v[key] = np.zeros_like(val)
      
    for key in params.keys():
      self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
      params[key] += self.v[key]
      