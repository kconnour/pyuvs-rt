import numpy as np
import matplotlib.pyplot as plt

ff = np.genfromtxt('/home/kyle/Downloads/iuvs_flatfield_133x19.txt')
w = np.linspace(200, 305, num=19)

for i in range(14):
    for j in range(10):
        if i != 13:
            plt.plot(w, ff[i*10 + j, :])
        else:
            plt.plot(w, ff[130, :])
            plt.plot(w, ff[131, :])
            plt.plot(w, ff[132, :])

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Average flatfield factor')
    plt.savefig(f'/home/kyle/ff{i}.png', dpi=300)
    plt.clf()