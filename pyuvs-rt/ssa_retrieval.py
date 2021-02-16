"""Find the UV single scattering albedo from the MY34 GDS
"""
# Built-in imports
import os
import time
from tempfile import mkdtemp

# 3rd-party imports
import numpy as np
from astropy.io import fits
from scipy import optimize
#import joblib

# Local imports
from pyuvs.data_contents import L1bDataContents
import disort
from pyRT_DISORT.controller import ComputationalParameters, ModelBehavior, \
    OutputArrays, UserLevel
from pyRT_DISORT.eos import eos_from_array
from pyRT_DISORT.flux import IncidentFlux, ThermalEmission
from pyRT_DISORT.observation import Angles, Wavelengths
from pyRT_DISORT.surface import Lambertian
from pyRT_DISORT.untested.aerosol import ForwardScatteringProperty, \
    ForwardScatteringPropertyCollection
from pyRT_DISORT.untested_utils.utilities.external_files import ExternalFile
from pyRT_DISORT.untested.aerosol_column import Column
from pyRT_DISORT.untested.vertical_profiles import Conrath
from pyRT_DISORT.untested.phase_function import TabularLegendreCoefficients
from pyRT_DISORT.untested.rayleigh import RayleighCo2
from pyRT_DISORT.untested.model_atmosphere import ModelAtmosphere
#from pyRT_DISORT.preprocessing.utilities.shared_array import SharedArray

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in external aerosol files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# New: I moved the relevant files into the tests directory. This way I can have
# files that change (like the atmosphere equation of state file) elsewhere
# while keeping my tests unchanged.

# Read in the atmosphere file
#tests_path = os.path.dirname(os.path.realpath(__file__))
aero_path = '/home/kyle/repos/pyuvs-rt/data'
eos_file = ExternalFile(os.path.join(aero_path, 'marsatm.npy'))

# Read in the dust scattering properties file
dust_file = ExternalFile(os.path.join(aero_path, 'dust_properties.fits'))

# Read in the dust phase function file
dust_phsfn_file = ExternalFile(os.path.join(aero_path,
                                            'dust_phase_function.fits'))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Observation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TODO: This is just a test file so I can see if the code works together. Loop over files
# TODO: Read in l1c files here too
dc = L1bDataContents('/media/kyle/Samsung_T5/IUVS_data/orbit07200/mvn_iuv_l1b_apoapse-orbit07252-muv_20180621T090628_v13_r01.fits.gz')

short_wav = dc.wavelengths[0, 0, :] / 1000   # TODO: these are centers, not edges
long_wav = (dc.wavelengths[0, 0, :] + 5) / 1000  # TODO: this is a hack
wavelengths = Wavelengths(short_wav, long_wav)
low_wavenumber = wavelengths.low_wavenumber
high_wavenumber = wavelengths.high_wavenumber

sza = dc.solar_zenith_angle
emission_angle = dc.emission_angle
phase_angle = dc.phase_angle

angles = Angles(sza, emission_angle, phase_angle)
mu = angles.mu
mu0 = angles.mu0
phi = angles.phi
phi0 = angles.phi0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the equation of state variables on a custom grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
z_boundaries = np.linspace(80, 0, num=20)    # Define the boundaries to use
# New: eos_from_array is a helper function to make a ModelEquationOfState class
# See eos.py for more info but this replaces ModelGrid (mostly I think the name
# is better)
model_eos = eos_from_array(eos_file.array, z_boundaries, 3.71, 7.3*10**-26)
temperatures = model_eos.temperature_boundaries
h_lyr = model_eos.scale_height_boundaries

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct aerosol/model properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# New: note that these will almost certainly change---either the classes
# themselves, or I ought to make functions that do some of this work
# Note: disort_multi ships with aerosol_dust.dat at 1.5 microns (index 10) of
# the forward scattering file. Also note that this has 14 particle sizes, not
# 13 like the phase function array I have
wavs = dust_file.array['wavelengths'].data
sizes = dust_file.array['particle_sizes'].data
c_ext = ForwardScatteringProperty(dust_file.array['primary'].data[:, :, 0],
                                  particle_size_grid=sizes,
                                  wavelength_grid=wavs)
c_sca = ForwardScatteringProperty(dust_file.array['primary'].data[:, :, 1],
                                  particle_size_grid=sizes,
                                  wavelength_grid=wavs)
dust_properties = ForwardScatteringPropertyCollection()
dust_properties.add_property(c_ext, 'c_extinction')
dust_properties.add_property(c_sca, 'c_scattering')

# Make a dust Conrath profile
conrath_profile = Conrath(model_eos, 10, 0.5)

# Define a smooth gradient of particle sizes. Here I'm making all particle sizes
# = 1.5 so I can compare with disort_multi (I don't know how to include a
# particle size gradient with it)
p_sizes = np.linspace(1.5, 1.5, num=len(conrath_profile.profile))

# Make a phase function. I'm allowing negative coefficients here
dust_phsfn = TabularLegendreCoefficients(
    dust_phsfn_file.array['primary'].data,
    dust_phsfn_file.array['particle_sizes'].data,
    dust_phsfn_file.array['wavelengths'].data)

# Make the new Column where wave_ref = 9.3 microns and column OD = 1
dust_col = Column(dust_properties, model_eos, conrath_profile.profile, p_sizes,
                  short_wav, 9.3, 1, dust_phsfn)

# Make Rayleigh stuff
n_moments = 1000
rco2 = RayleighCo2(short_wav, model_eos, n_moments)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model = ModelAtmosphere()
# Make tuples of (dtauc, ssalb, pmom) for each constituent
dust_info = (dust_col.total_optical_depth, dust_col.scattering_optical_depth,
             dust_col.scattering_optical_depth * dust_col.phase_function)
rayleigh_info = (rco2.scattering_optical_depths, rco2.scattering_optical_depths,
                 rco2.phase_function)  # This works since scat OD = total OD

# Add dust and Rayleigh scattering to the model
model.add_constituent(dust_info)
model.add_constituent(rayleigh_info)

# Once everything is in the model, compute the model. Then, slice off the
# wavelength dimension since DISORT can only handle 1 wavelength at a time
model.compute_model()
optical_depths = model.hyperspectral_total_optical_depths[:, 1]
ssa = model.hyperspectral_total_single_scattering_albedos[:, 1]
polynomial_moments = model.hyperspectral_legendre_moments[:, :, 1]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the size of the computational parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Semi-new: this class just holds parameters DISORT wants. It's kinda useless by
# itself but a valuable input into the upcoming classes
n_layers = model_eos.n_layers
n_streams = 16
n_umu = 1
n_phi = len(phi)
n_user_levels = 81
cp = ComputationalParameters(
    n_layers, n_moments, n_streams, n_phi, n_umu, n_user_levels)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make misc variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# New: I split the old "control" classes into additional classes that I think
# are more accurately named and grouped. I think many of these variables in
# DISORT are horribly named. For clarity, I included the "DISORT" name as the
# variable name, and the name I prefer as the property
# (i.e. fisot = isotropic_flux). There's really no reason to define any of these
# variables here---you can just put them directly into the disort call---but I
# thought it might be helpful.

# Semi-new: another note, that many of these variables take a boolean or float
# value. I made them optional, and use default values that disort_mulit uses
incident_flux = IncidentFlux()
fbeam = incident_flux.beam_flux
fisot = incident_flux.isotropic_flux

te = ThermalEmission()
plank = te.thermal_emission
btemp = te.bottom_temperature
ttemp = te.top_temperature
temis = te.top_emissivity

mb = ModelBehavior()
accur = mb.accuracy
deltamplus = mb.delta_m_plus
dopseudosphere = mb.do_pseudo_sphere
header = mb.header
ibcnd = mb.incidence_beam_conditions
onlyfl = mb.only_fluxes
prnt = mb.print_variables
radius = mb.radius
usrang = mb.user_angles
usrtau = mb.user_optical_depths

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the output arrays
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
oa = OutputArrays(cp)
albmed = oa.albedo_medium
flup = oa.diffuse_up_flux
rfldn = oa.diffuse_down_flux
rfldir = oa.direct_beam_flux
dfdt = oa.flux_divergence
uu = oa.intensity
uavg = oa.mean_intensity
trnmed = oa.transmissivity_medium

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Optical depth output structure
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# I made this into its own class to handle utau. This one singular variable is
# an absolute nightmare and DISORT should be tweaked to get rid of it, but
# you're not paying me to discuss the bad decisions that went into making this
utau = UserLevel(cp, mb).optical_depth_output

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Surface treatment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# New: I made an abstract Surface class, and all surfaces you'd use inherit from
# it. The idea is that surface makes arrays of 0s for rhou, rhoq, bemst, etc. In
# the special case of a Lambertian surface, these are fine inputs to DISORT
# since it ignores these arrays when LAMBER=True. Otherwise, these arrays get
# populated when you call disobrdf. For instance, if you instantiate HapkeHG2,
# it calls disobrdf in the constructor which populates the surface arrays. The
# benefit to the user is that all classes derived from Surface will have the
# same properties.
lamb = Lambertian(0.5, cp)   # albedo = 0.5
albedo = lamb.albedo
lamber = lamb.lambertian
rhou = lamb.rhou
rhoq = lamb.rhoq
bemst = lamb.bemst
emust = lamb.emust
rho_accurate = lamb.rho_accurate

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
    disort.disort(usrang, usrtau, ibcnd, onlyfl, prnt, plank, lamber,
                  deltamplus, dopseudosphere, optical_depths, ssa,
                  polynomial_moments, temperatures, low_wavenumber,
                  high_wavenumber, utau, mu0, phi0, mu, phi, fbeam, fisot,
                  albedo, btemp, ttemp, temis, radius, h_lyr, rhoq, rhou,
                  rho_accurate, bemst, emust, accur, header, rfldir,
                  rfldn, flup, dfdt, uavg, uu, albmed, trnmed)

print(uu[0, 0, 0])   # shape: (1, 81, 1)
# This gives          0.041567463
# disort_multi gives  0.0415661298
# I'm running ./disort_multi -dust_conrath 0.5, 10 -dust_phsfn 98 -NSTR 16 < testInput.txt
# testInput.txt is: 1, 0.5, 10, 30, 50, 40, 20, 1, 0, 0
# And dust_phsfn has 65 moments at 1.5 micron (size) and 9.3 microns(wavelength)




'''# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These are all calculations I have to re-do for each time I try a new parameter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# THIS USES MULTIPROCESSING MODULE
# Make an array in shared memory
junkThing = np.zeros((len(szas), len(wavs)), dtype=np.float)
shm = shared_memory.SharedMemory(name='shared_answer', create=True, size=junkThing.nbytes)
answer = np.ndarray(junkThing.shape, dtype=junkThing.dtype, buffer=shm.buf)
answer[:] = junkThing[:]

# THIS USES JOBLIB MODULE
#filename = os.path.join(mkdtemp(), 'myNewFile.dat')
#answer = np.memmap(filename, dtype=np.float, shape=(len(szas), len(wavs)), mode='w+')

# Use my code to create an array in shared memory
sa = SharedArray((len(szas), len(wavs)))
answer = sa.array


def myModel(guess, pixel):
    dust_od = guess[0]
    ice_od = guess[1]
    if dust_od <= 0 or ice_od <= 0:
        return 99999
    # The column will change everytime I use a different OD
    dust_column = Column(dust, lay, dust_conrath, np.array([1]), np.array([dust_od]))
    ice_column = Column(ice, lay, ice_profile, np.array([2]), np.array([ice_od]))

    # The phase functions change as columns change
    dust_phase = TabularLegendreCoefficients(dust_column, dust_phase_function.array,
                                             particle_sizes=dust_phase_radii.array, wavelengths=dust_phase_wavs.array)
    ice_phase = TabularLegendreCoefficients(ice_column, ice_coeff.array)

    # Make the model
    model = ModelAtmosphere()
    dust_info = (dust_column.hyperspectral_total_optical_depths, dust_column.hyperspectral_scattering_optical_depths,
                 dust_phase.coefficients)
    ice_info = (ice_column.hyperspectral_total_optical_depths, ice_column.hyperspectral_scattering_optical_depths,
                ice_phase.coefficients)

    # Add everything to the model
    model.add_constituent(dust_info)
    model.add_constituent(ice_info)
    model.add_constituent(rayleigh_info)

    # Once everything is in the model, compute the model
    model.compute_model()
    optical_depths = model.hyperspectral_total_optical_depths
    ssa = model.hyperspectral_total_single_scattering_albedos
    polynomial_moments = model.hyperspectral_legendre_moments

    #existing_shm = shared_memory.SharedMemory(name='shared_answer')
    #shared_array = np.ndarray((len(szas), len(wavs)), dtype=np.float, buffer=existing_shm.buf)
    for w in range(len(wavs)):
        # The 0s here are just me testing this on a single pixel
        junk, junk, junk, junk, junk, uu, junk, junk = disort.disort(usrang, usrtau, ibcnd, onlyfl, prnt, plank,
             lamber, deltamplus, do_pseudo_sphere, optical_depths[:, w], ssa[:, w], polynomial_moments[:, :, w], temperatures,
             low_wavenumber[w], high_wavenumber[w], utau, umu0[pixel], phi0, umu[pixel], phi[pixel], fbeam, fisot, albedo,
             surface_temp, top_temp, top_emissivity, planet_radius, h_lyr, hapke.rhoq, hapke.rhou, hapke.rho_accurate[:, pixel],
             hapke.bemst, hapke.emust, accur, header, direct_beam_flux, diffuse_down_flux,
             diffuse_up_flux, flux_divergence, mean_intensity, intensity[:, :, pixel], albedo_medium, transmissivity_medium)
        # This is what I'd do for multiprocessing
        #shared_array[pixel, w] = uu[0, 0]
        answer[pixel, w] = uu[0, 0]


def calculate_model_difference(guess, pixel):
    # This is just a sample spectrum I took... each pixel should have its own
    target = np.array([0.784799E-01, 0.753058E-01, 0.698393E-01, 0.654867E-01, 0.605224E-01, 0.572824E-01, 0.534369E-01,
                       0.496474E-01, 0.464914E-01, 0.427855E-01, 0.429494E-01, 0.457046E-01, 0.491979E-01, 0.495246E-01,
                       0.457136E-01, 0.433431E-01, 0.483898E-01, 0.504774E-01, 0.457426E-01])
    test_run = myModel(guess, pixel)
    return np.sum((test_run - target)**2)


def do_optimization(guess, pixel):
    return optimize.minimize(calculate_model_difference, np.array(guess), pixel, method='Nelder-Mead').x


# Without this pool won't cooperate
if __name__ == '__main__':
    t0 = time.time()
    n_cpus = 8  # mp.cpu_count()
    pixel_inds = np.linspace(0, len(szas)-1, num=len(szas), dtype='int')
    #iterable = [[[0.8, 0.2], f] for f in pixel_inds]

    #m = joblib.Parallel(n_jobs=-2, prefer='processes')(joblib.delayed(myModel)([0.8, 0.2], f) for f in pixel_inds)
    joblib.Parallel(n_jobs=-2, prefer='processes')(joblib.delayed(myModel)([0.8, 0.2], f) for f in pixel_inds)
    print(np.amax(answer))
    #np.save('/home/kyle/bestArray.npy', answer)
    #sa.delete()
    t1 = time.time()
    print(t1 - t0)'''
