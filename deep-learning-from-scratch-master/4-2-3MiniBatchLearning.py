# ミニバッチ学習

import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
  load_mnist(normalize=True, one_hot_label=True)
  # normilize=Trueで入力画像を0.0~1.0の値に正規化する
  # one_hot_label=Trueで正解となるラベルが1,それ以外が0の配列。Falseのときは7, 2といった正解のラベルがそのまま格納される

# それぞれのデータの形状を出力
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10) 訓練画像が60000枚


train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

def cross_entropy_error(y, t): 
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  batch_size = y.shape[0]
  return -np.sum( t * np.log(y + 1e-7) ) / batch_size # 教師データがone-hot表現のとき
  return -np.sum( np.log( y[np.arange(batch_size), t] + 1e-7 ) ) / batch_size # 教師データがラベルのとき