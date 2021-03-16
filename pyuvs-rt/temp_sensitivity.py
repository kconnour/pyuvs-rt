import numpy as np


f = np.load('/home/kyle/ssa_retrievals/retrieved_ssa_1-5_microns_conrath_10_01-phasefunctionhack-50temp.npy')
g = np.load('/home/kyle/ssa_retrievals/retrieved_ssa_1-5_microns_conrath_10_01-phasefunctionhack-minus50temp.npy')

fm = np.nanmean(f, axis=0)
gm = np.nanmean(g, axis=0)

print(fm / gm - 1)
'''Conclusion: the absolute magnitude of the temperature doesn't matter.
This makes sense cause T is used to compute the column density, which is 
just a scaling factor when computing the OD.'''

h = np.load('/home/kyle/ssa_retrievals/retrieved_ssa_1-5_microns_conrath_10_01-phasefunctionhack-realTemp.npy')
hm = np.nanmean(h, axis=0)
print(fm / hm - 1)
'''Conclusion: using a real profile doesn't really change the answer much
compared to using the default profile.'''