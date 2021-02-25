Things to do first:
- Interpolate Gale Crater ODs as a function of Ls
- Get pixels over Gale Crater where OD > 5, SZA<70, emission<70. All these
  should be np.arrays
    - Orbit (N)
    - Filename (N)
    - Integration (N)
    - Position (N)
    - Ls (N)
    - Lat (N)
    - Lon (N)
    - SZA (N)
    - Emission (N)
    - Phase (N)
    - LT (N)
    - I/F (N, 19)
    - Uncertainty (N, 19)
    

Things to explore
- I should try many surfaces to see if it matters
- I should start with the Montabone profiles but explore GCMs, assimilated
profiles, etc.

Learn
- Find previous UV dust SSA
- See Guzewich and Lemmon for particle sizes to start with

Don't do
- There's no sense in using pseudo spherical correction
- Don't coadd pixels. Stats say they should average out

it's ok to set the extinction cross section to 1 and fit the scattering cross
section as a proxy for SSA