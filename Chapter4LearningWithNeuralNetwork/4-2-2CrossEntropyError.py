# 交差エントロピー誤差の実装

import numpy as np

def cross_entropy_error(y, t): # yとtはnumpy配列
  delta = 1e-7 
  return -np.sum(t * np.log(y + delta))
  # 微少な値deltaにより、np.log(0) == -inf となり、計算が先に進まなくなるのを防止



