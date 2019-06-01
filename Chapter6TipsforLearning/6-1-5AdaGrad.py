# ニューラルネットワークでは学習係数が大きすぎると学習が発散し、小さすぎると時間がかかるため、適切な係数設定が必要
# この学習係数に関する有効なテクニックとして、学習係数の減衰(learning rate decay)がある
# 学習全体ではなく、個別のニューロンに対して学習係数を減衰させていく手法がAdaGrad

class AdaGrad:
  def __init__(self, lr=0.01):
    self.lr = lr
    self.h = None
  
  def update(self, params, grads):
    if self.h is None:
      self.h = {}
      for key, val in params.items():
        self.h[key] = np.zeros_like(val)
            
    for key in params.keys():
      self.h[key] += grads[key] * grads[key]
      params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)