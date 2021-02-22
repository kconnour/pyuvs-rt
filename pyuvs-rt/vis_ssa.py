import matplotlib.pyplot as plt
import numpy as np

a = np.load('/home/kyle/retrieved_ssa.npy')
b = np.linspace(0, 18, num=19)
for i in [50, 51, 52]:
    for j in [50, 51, 52, 53, 54, 55, 56]:
        if i == 50:
            c = 'r'
        if i == 51:
            c = 'g'
        if i == 52:
            c = 'b'
        plt.scatter(b, a[i, j, :], color=c)

plt.ylim(0.6, 0.75)
plt.savefig('/home/kyle/ssa.png', dpi=150)
