# James Rogers
# 3-3-2024
# Visualization of table I in 1998 Feldman-Cousins paper

import numpy as np
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y1 = [0.030, 0.106, 0.185, 0.216, 0.189, 0.132, 0.077, 0.039, 0.017, 0.007, 0.002, 0.001]
y2 = [0.050, 0.149, 0.224, 0.224, 0.195, 0.175, 0.161, 0.149, 0.140, 0.132, 0.125, 0.119]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8))

ax1.plot(x, y1, '--', 
	color='blue', marker='o', markersize=8, label='P(n|$\mu$=0.5)')
ax1.plot(x, y2, '--', 
	color='red', marker='o', markersize=8, label='P(n|$\mu_{best}$)')

ax2.plot(x, np.divide(y1,y2), '--', 
	color='black', marker='o', markersize=8, label='P(n|$\mu$=0.5)/P(n|$\mu_{best}$)')

#ax1.set_xlabel('n')
ax1.set_ylabel('P [1]')
ax1.set_title('Poisson process with Background')
ax1.legend()
ax1.set_xbound(np.min(x)-0.5, np.max(x)+0.5)

ax2.set_xlabel('n')
ax2.set_ylabel('R [1]')
#ax2.set_title('Ratio')
ax2.legend()
ax2.set_xbound(np.min(x)-0.5, np.max(x)+0.5)

plt.tight_layout()
plt.show()
