# シグモイド関数の実装と表示

import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
# Numpyのブロードキャストにより、スカラーとNumpy配列の演算はスカラーとNumpy配列の要素同士で行われる

# >>> import numpy as np
# >>> t = np.array([1.0, 2.0, 3.0])
# >>> 1.0 + t
# array([2., 3., 4.])
# >>> 1.0 / t
# array([1.        , 0.5       , 0.33333333])

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y軸の範囲の指定
plt.show()