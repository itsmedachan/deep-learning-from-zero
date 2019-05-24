# ReLU関数の実装
# 0以下だったら0, 正だったらその値を返す

def relu(x):
  return np.maximum(0, x)
# maximumは、大きい方の値を返す