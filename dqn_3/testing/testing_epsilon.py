import time
import numpy as np
import matplotlib.pyplot as plt
import math

FEF = int(1e6)
IE = 1
FE = 0.1
s = []

start = time.time()

slope = (1.0-0.1)/(0.0-1e6)

print(f"slope: {slope}")

for i in range(int(2.1e6)):
    # for i in range(int(100)):
    # e = np.interp(i, [0, FEF], [IE, FE])

    # e = FE + (IE - FE) * math.exp(-1 * i / FEF)

    e = slope*i+1 if i <= 1e6 else 0.1

    # e = e*slope

    # s.append([i, e])
    s.append(e)

end = time.time()

print(f"time: {end-start}")

print(len(s))

plt.plot(s)
plt.show()
