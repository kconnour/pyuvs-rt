import numpy as np


f = np.load('/home/kyle/ssa_retrievals/retrieved_ssa_1-5_microns_conrath_10_01-phasefunctionhack.npy')
g = np.load('/home/kyle/ssa_retrievals/retrieved_ssa_1-5_microns_conrath_10_01-phasefunctionhack-realTemp-clancyLamberSurface.npy')
h = np.load('/home/kyle/ssa_retrievals/retrieved_ssa_1-5_microns_conrath_10_01-phasefunctionhack-realTemp-HighLamberSurface.npy')  # 10% to 15% blue to red
i = np.load('/home/kyle/ssa_retrievals/retrieved_ssa_1-5_microns_conrath_10_01-phasefunctionhack-realTemp-ExtremelyHighLamberSurface.npy')  # 94% to 99% blue to red
j = np.load('/home/kyle/ssa_retrievals/retrieved_ssa_1-5_microns_conrath_10_01-phasefunctionhack-realTemp-LowHapkeSurface.npy')  # The values in Wolff 2009
l = np.load('/home/kyle/ssa_retrievals/retrieved_ssa_1-5_microns_conrath_10_01-phasefunctionhack-realTemp-wolffPressure-LowHapkeSurface.npy')  # same as j but 6.1 mbar (the other was 6.7 mbar)

fm = np.nanmean(f, axis=0)
gm = np.nanmean(g, axis=0)
hm = np.nanmean(h, axis=0)
im = np.nanmean(i, axis=0)
jm = np.nanmean(j, axis=0)
lm = np.nanmean(l, axis=0)
print(lm)
print(jm)

print(fm / gm - 1)
print(hm / gm - 1)
print(im / gm - 1)
