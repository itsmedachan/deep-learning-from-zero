def AND_theta(x1, x2):
  w1, w2, theta = 0.5, 0.5, 0.7
  tmp = x1*w1 + x2*w2
  if tmp <= theta:
    return 0
  elif tmp > theta:
    return 1

print(AND_theta(0, 0)) # 0
print(AND_theta(0, 1)) # 0
print(AND_theta(1, 0)) # 0
print(AND_theta(1, 1)) # 1

# --------------------
# 0 ( x1*w1 + x2*w2 <= theta )
# 1 ( x1*w1 + x2*w2 > theta )
# w1,w2:重み、theta:閾値

# の表記を今後のため以下に変更

# 0 ( b + x1*w1 + x2*w2 <= 0 )
# 1 ( b + x1*w1 + x2*w2 > 0 )
# w1,w2:重み、b:バイアス(b = -theta)
# --------------------

import numpy as np
x = np.array([0,1]) # 入力
w = np.array([0.5, 0.5]) # 重み
b = -0.7 # バイアス

print(w*x)
# [0.  0.5]
print(np.sum(w*x))
# 0.5
print(np.sum(w*x) + b)
# -0.19999999999999996 #およそ-0.2(浮動小数点数による演算誤差)

# thetaではなくバイアス(-theta)を用いてANDゲートを再実装
def AND(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.7
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

# 重み→入力信号の重要度をコントロールするパラメータ
# バイアス→ニューロンの発火のしやすさ
# 学習=パラメーターを変更し、正解に近づけていくこと

def NAND(x1, x2):
  x = np.array([x1, x2])
  w = np.array([-0.5, -0.5])
  b = 0.7 # 重みとバイアスの符号をANDと反転させるだけ
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

def OR(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.2
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1


# ---------------------
# 単層パーセプトロンでAND,NAND,ORゲートの実装できた
# XOR(排他的論理和)ゲートは、上記3つのパーセプトロンを組み合わせることで実装可能
# →多層パーセプトロン
# ---------------------

def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2) # NANDゲートとORゲートの出力結果をANDゲートの入力にしたパーセプトロン
  return y

print(XOR(0, 0)) # 0
print(XOR(0, 1)) # 1
print(XOR(1, 0)) # 1
print(XOR(1, 1)) # 0