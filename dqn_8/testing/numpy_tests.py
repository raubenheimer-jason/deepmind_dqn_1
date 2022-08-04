import numpy as np
from collections import deque

s = deque(maxlen=4)

# for i in range(4):
#     # print(i)
#     n = np.full(shape=(5, 5), fill_value=i)
#     s.append(n)

# print(s)

ret = np.stack(s)

print(ret.shape)