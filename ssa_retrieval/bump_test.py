"""Test the sensitivity of fixing the bump
"""
# Built-in imports
import os
import time
from tempfile import mkdtemp
import multiprocessing as mp
import matplotlib.pyplot as plt

# 3rd-party imports
import numpy as np
from astropy.io import fits
from scipy import optimize

# Local imports
import disort
from pyRT_DISORT.observation import Angles, Spectral
from pyRT_DISORT.eos import Hydrostatic
from pyRT_DISORT.rayleigh import RayleighCO2
from pyRT_DISORT.aerosol import Conrath, ForwardScattering, OpticalDepth, \
    TabularLegendreCoefficients
from pyRT_DISORT.atmosphere import Atmosphere
from pyRT_DISORT.controller import ComputationalParameters, ModelBehavior
from pyRT_DISORT.radiation import IncidentFlux, ThermalEmission
from pyRT_DISORT.output import OutputArrays, OutputBehavior, UserLevel
from pyRT_DISORT.surface import Surface
from gale_crater import GaleOpticalDepth

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Observation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#file = '/home/kyle/Downloads/mvn_iuv_l1c_apoapse-orbit08307-muv_20190101T103348_v13_r01.fits'   # top of scan -2
file = '/home/kyle/Downloads/mvn_iuv_l1c_apoapse-orbit08307-muv_20190101T100914_v13_r01.fits'   # top of scan 2

hdul = fits.open(file)

reflectance = hdul['primary'].data
wavelengths = (np.load('/home/kyle/iuvs_wavelengths.npy') / 1000)
solar_zenith_angle = hdul['solar_zenith_angle'].data
emission_angle = hdul['emission_angle'].data
phase_angle = hdul['phase_angle'].data
ls = hdul['ls'].data
latitude = hdul['latitude'].data
longitude = hdul['longitude'].data

bin_width = 0.00281   # microns

# Define spectral info
short_wav = (wavelengths - bin_width)
long_wav = (wavelengths + bin_width)
spectral = Spectral(short_wav, long_wav)

# Define angles
angles = Angles(solar_zenith_angle, emission_angle, phase_angle)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Interpolate Curiosity ODs to IUVS pixels
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pixel_od = GaleOpticalDepth().interpolate_tau(ls)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the equation of state variables on a custom grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
aux_files_path = '/home/kyle/repos/pyuvs-rt/data'
z_boundaries = np.linspace(80, 0, num=15)    # Define the boundaries to use
eos_file = np.flipud(np.load(os.path.join(aux_files_path, 'marsatm.npy')))

# Force P_surface to be 6.1 mbar
eos_file[:, 1] *= 610 / eos_file[-1, 1]
mass = 7.3 * 10**-26
gravity = 3.7

hydro = Hydrostatic(eos_file[:, 0], eos_file[:, 1], eos_file[:, 2],
                    z_boundaries, mass, gravity)

###########################
# Aerosol things
###########################
# Rayleigh scattering
rco2 = RayleighCO2(wavelengths, hydro.column_density)
rayleigh_info = (rco2.optical_depth, rco2.single_scattering_albedo,
                 rco2.phase_function)

# Dust vertical profile
z_midpoint = ((z_boundaries[:-1] + z_boundaries[1:]) / 2)
q0 = 1
H = 10
nu = 0.01

conrath = Conrath(z_midpoint, q0, H, nu)
dust_profile = conrath.profile

# Read in the dust properties
dust_fsp = os.path.join(aux_files_path, 'dust_properties.fits')
hdul = fits.open(dust_fsp)
cext = hdul['primary'].data[:, :, 0]
csca = hdul['primary'].data[:, :, 1]
fsp_wavs = hdul['wavelengths'].data
fsp_psizes = hdul['particle_sizes'].data

# TODO: run my code with several of these
pgrad = np.linspace(1.5, 1.5, num=len(z_boundaries) - 1)
wave_ref = 0.88

# Read in the dust phase function
dust_pf = os.path.join(aux_files_path, 'dust_phase_function.fits')
hdul = fits.open(dust_pf)
phsfn = hdul['primary'].data
pf_wavs = hdul['wavelengths'].data
pf_psizes = hdul['particle_sizes'].data

# Use constant properties over the MUV region
fsp_wavs[1] = 0.439
pf_wavs[1] = 0.439

# Make the misc things
cp = ComputationalParameters(hydro.n_layers, 65, 16, 1, 1, 80)
mb = ModelBehavior()
flux = IncidentFlux()
te = ThermalEmission()
ob = OutputBehavior()
ulv = UserLevel(cp.n_user_levels)

# Surface treatment
# Use Todd Clancy's surface
clancy_lamber = np.interp(wavelengths, np.linspace(0.2, 0.33, num=100),
                          np.linspace(0.01, 0.015, num=100))
lamb = [Surface(w, cp.n_streams, cp.n_polar, cp.n_azimuth, ob.user_angles,
                ob.only_fluxes) for w in clancy_lamber]

# Make a Lambert surface. If I want a Hapke surface, that needs to be done in
# retrieve_ssa
for l in lamb:
    l.make_lambertian()


def retrieve_od(od_guess, pixel_index, target_spectrum):
    # Skip the retrieval if the pixel quantities aren't good
    answer = np.nan
    if solar_zenith_angle[pixel_index] > 72 or emission_angle[pixel_index] > 72:
        print(f'skipping pixel {pixel_index}')
        return answer

    fs = ForwardScattering(csca, cext, fsp_psizes, fsp_wavs,
                           pgrad, wavelengths, wave_ref)
    fs.make_nn_properties()
    tlc = TabularLegendreCoefficients(phsfn, pf_psizes, pf_wavs, pgrad,
                                      wavelengths)
    tlc.make_nn_phase_function()

    def fit_od(test_od):
        # Trap the guess
        if not 0 <= test_od <= 2:
            return 9999999

        od = OpticalDepth(dust_profile, hydro.column_density, fs.extinction,
                          test_od)

        dust_info = (od.total, fs.single_scattering_albedo, tlc.phase_function)
        model = Atmosphere(rayleigh_info, dust_info)
        print(test_od)
        print(model.single_scattering_albedo)

        od_holder = np.zeros(19)
        for wav_index in range(19):
            oa = OutputArrays(cp.n_polar, cp.n_user_levels, cp.n_azimuth)


            rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
                disort.disort(ob.user_angles, ob.user_optical_depths,
                              ob.incidence_beam_conditions, ob.only_fluxes,
                              mb.print_variables, te.thermal_emission,
                              lamb[wav_index].lambertian,
                              mb.delta_m_plus, mb.do_pseudo_sphere,
                              model.optical_depth[:, wav_index],
                              model.single_scattering_albedo[:, wav_index],
                              model.legendre_moments[:, :, wav_index],
                              hydro.temperature, spectral.low_wavenumber,
                              spectral.high_wavenumber, ulv.optical_depth_output,
                              angles.mu0[pixel_index], angles.phi0[pixel_index],
                              angles.mu[pixel_index], angles.phi[pixel_index],
                              flux.beam_flux, flux.isotropic_flux,
                              lamb[wav_index].albedo, te.bottom_temperature,
                              te.top_temperature, te.top_emissivity,
                              mb.radius, hydro.scale_height, lamb[wav_index].rhoq,
                              lamb[wav_index].rhou, lamb[wav_index].rho_accurate,
                              lamb[wav_index].bemst, lamb[wav_index].emust,
                              mb.accuracy, mb.header, oa.direct_beam_flux,
                              oa.diffuse_down_flux, oa.diffuse_up_flux,
                              oa.flux_divergence, oa.mean_intensity, oa.intensity,
                              oa.albedo_medium, oa.transmissivity_medium)
            od_holder[wav_index] = uu[0, 0, 0]
        return np.sum((od_holder - target_spectrum)**2)

    fitted_od = optimize.minimize(fit_od, np.array([od_guess]),
                                  method='Nelder-Mead').x
    return fitted_od


t0 = time.time()
ans = np.zeros(13)
for i in range(13):
    ref = np.copy(reflectance[0, i * 10, :])
    ref[[-2, -3, -6, -7, -11]] *= 0.9
    od = retrieve_od(0.3, (0, i * 10), ref)
    ans[i] = od

ans0 = np.zeros(13)
for i in range(13):
    ref = reflectance[0, i * 10, :]
    od = retrieve_od(0.3, (0, i * 10), ref)
    ans0[i] = od
t1 = time.time()
print(ans0 - ans)
print((ans0 - ans) / ans0)
print(f'It took {t1 - t0} seconds to get the dust OD.')
