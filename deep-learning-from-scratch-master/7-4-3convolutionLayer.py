import sys, os
sys.path.append(os.pardir)
from common.util import im2col
import numpy as np


x1 = np.random.rand(1, 3, 7, 7)
# np.random.rand(データの数, チャンネル数, 高さ, 横幅)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)
# (9, 75)

x2 = np.random.rand(10, 3, 7, 7) # 10個のデータ
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)
# (90, 75)


class Convolution:
  def __init__(self, W, b, stride=0, pad=0):
    self.W = W
    self.b = b
    self.stride = stride
    self.pad = pad
  
  def forward(self, x):
    FN, C, FH, FW, = self.W.shape
    N, C, H, W = x.shape
    out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
    out_w = int(1 + (H + 2 * self.pad - FW) / self.stride)

    col = im2col(x, FH, FW, self.stride, self.pad)
    col_W = self.W.reshape(FN, -1).T # フィルターの展開
    out = np.dot(col, col_W) + self.b

    out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
    return out

