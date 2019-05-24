# ステップ関数の実装

import numpy as np

# ステップ関数
def step_function_reject_numpy(x):
  if x > 0:
    return 1
  else:
    return 0

# 上記の引数xはNumpy配列を受け取らないので以下のように変更
def step_function(x):
  y = x > 0
  return y.astype(np.int)

# 以上により、Numpyの不等号による演算を行う
# >>> import numpy as np
# >>> x = np.array([-1.0, 1.0, 2.0])
# >>> x
# array([-1.,  1.,  2.])
# >>> y = x > 0
# >>> y
# array([False,  True,  True])
# >>> y = y.astype(np.int)
# >>> y
# array([0, 1, 1])

