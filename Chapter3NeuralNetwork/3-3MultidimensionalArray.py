# 多次元配列の演算

# 一次元配列の復習
>>> import numpy as np
>>> A = np.array([1, 2, 3, 4])
>>> print(A)
[1 2 3 4]
>>> np.ndim(A)
1
>>> A.shape
(4,) # 結果がタプルになっている
>>> A.shape[0]
4

# 二次元配列の作成
>>> B = np.array([[1, 2], [3, 4], [5, 6]])
>>> print(B)
[[1 2]
 [3 4]
 [5 6]]
>>> np.ndim(B)
2
>>> B.shape
# (3, 2) # 最初の次元に3つの要素、次の次元に2つの要素があるという意

# 行列の積
>>> A = np.array([[1, 2, 3], [4, 5, 6]])
>>> A.shape
(2, 3)
>>> B = np.array([[1, 2], [3, 4], [5, 6]])
>>> B.shape
(3, 2)
>>> np.dot(A, B)
array([[22, 28],
       [49, 64]])
# 行列Aの1次元めの要素数(列数)(3)と、行列Bの0次元めの要素数(行数)(3)を同じ値にする必要あり

# 要素数が合わず積が計算できない場合はエラーを吐く
>>> C = np.array([[1, 2], [3, 4]])
>>> C.shape
(2, 2)
>>> A.shape
(2, 3)
>>> np.dot(A, C)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)

# Aが二次元配列、Bが一次元配列でも、対応する次元の要素数を一致させる
>>> A = np.array([[1, 2], [3, 4], [5, 6]])
>>> A.shape
(3, 2)
>>> B = np.array([7, 8])
>>> B.shape
(2,)
>>> np.dot(A, B)
array([23, 53, 83])