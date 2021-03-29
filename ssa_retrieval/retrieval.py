"""Find the UV single scattering albedo from the MY34 GDS
"""
# Built-in imports
import os
import time
from tempfile import mkdtemp
import multiprocessing as mp

# 3rd-party imports
import numpy as np
from astropy.io import fits
from scipy import optimize

# Local imports
import disort
from pyRT_DISORT.controller import ComputationalParameters, ModelBehavior, \
    OutputArrays, UserLevel
from pyRT_DISORT.eos import eos_from_array
from pyRT_DISORT.flux import IncidentFlux, ThermalEmission
from pyRT_DISORT.observation import Angles, Wavelengths
from pyRT_DISORT.surface import Lambertian, HapkeHG2, HapkeHG2Roughness
from pyRT_DISORT.untested.aerosol import ForwardScatteringProperty, \
    ForwardScatteringPropertyCollection
from pyRT_DISORT.untested_utils.utilities.external_files import ExternalFile
from pyRT_DISORT.untested.aerosol_column import Column
from pyRT_DISORT.untested.vertical_profiles import Conrath
from pyRT_DISORT.untested.phase_function import TabularLegendreCoefficients
from pyRT_DISORT.untested.rayleigh import RayleighCo2
from pyRT_DISORT.untested.model_atmosphere import ModelAtmosphere
from ssa_retrieval.gale_crater import OpticalDepth

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in external aerosol files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in the atmosphere file
aero_path = '/home/kyle/repos/pyuvs-rt/data'
#eos_file = ExternalFile(os.path.join(aero_path, 'marsatm.npy'))
eos_file = np.load(os.path.join(aero_path, 'marsatm.npy'))

# Read in the dust scattering properties file
dust_file = ExternalFile(os.path.join(aero_path, 'dust_properties.fits'))

# Read in the dust phase function file
dust_phsfn_file = ExternalFile(os.path.join(aero_path,
                                            'dust_phase_function.fits'))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the equation of state variables on a custom grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
z_boundaries = np.linspace(80, 0, num=20)    # Define the boundaries to use
model_eos = eos_from_array(eos_file, z_boundaries, 3.71, 7.3*10**-26)
temperatures = model_eos.temperature_boundaries
h_lyr = model_eos.scale_height_boundaries

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct the immutable aerosol properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
wavs = dust_file.array['wavelengths'].data
sizes = dust_file.array['particle_sizes'].data

# Hack my code like a scientist
wavs[1] = 0.439

# Hack my code like a scientist
dustwavs = dust_phsfn_file.array['wavelengths'].data
dustwavs[1] = 0.439

# Make a phase function. I'm allowing negative coefficients here
dust_phsfn = TabularLegendreCoefficients(
    dust_phsfn_file.array['primary'].data,
    dust_phsfn_file.array['particle_sizes'].data,
    dustwavs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make misc variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# Observation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
file = '/home/kyle/repos/pyuvs-rt/ssa_files/gale_pixels.fits'

hdul = fits.open(file)
reflectance = hdul['reflectance'].data
n_wavelengths = 19

bin_width = 2.81   # nm
wavelengths = hdul['wavelengths'].data

short_wav = (wavelengths - bin_width) / 1000   # convert to microns
long_wav = (wavelengths + bin_width) / 1000
wavelengths = Wavelengths(short_wav, long_wav)
low_wavenumber = wavelengths.low_wavenumber
high_wavenumber = wavelengths.high_wavenumber

solar_zenith_angle = hdul['sza'].data
emission_angle = hdul['ea'].data
phase_angle = hdul['pa'].data
ls = hdul['ls'].data

angles = Angles(solar_zenith_angle, emission_angle, phase_angle)
mu = angles.mu
mu0 = angles.mu0
phi = angles.phi
phi0 = angles.phi0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Interpolate Curiosity ODs to IUVS pixels
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pixel_od = OpticalDepth().interpolate_tau(ls)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make a shared array
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
memmap_filename = os.path.join(mkdtemp(), 'myNewFile.dat')
retrieved_ssas = np.memmap(memmap_filename, dtype=float, shape=reflectance.shape,
                           mode='w+')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the size of the computational parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n_layers = model_eos.n_layers
n_streams = 16
n_umu = 1
n_phi = 1
n_user_levels = 81
n_moments = 1000
cp = ComputationalParameters(
    n_layers, n_moments, n_streams, n_phi, n_umu, n_user_levels)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Surface treatment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#lamb = Lambertian(0.02, cp)
#hapke_w = np.interp(wavelengths.short_wavelengths[0, :], [258, 320], [0.07, 0.095])
hapke_w = np.interp(wavelengths.short_wavelengths[0, :], np.linspace(0.2, 0.33, num=100), np.linspace(0.01, 0.015, num=100))
hapke = [Lambertian(w, cp) for w in hapke_w]
#hapke = [HapkeHG2Roughness(0.01, cp, mb, incident_flux, angles, 0.8, 0.06, w, 0.3, 0.45, 20) for w in hapke_w]
#lamb = HapkeHG2Roughness(0.01, cp, mb, incident_flux, angles, 0.8, 0.06, 0.08, 0.3, 0.45, 20)
#albedo = lamb.albedo
#lamber = lamb.lambertian
#rhou = lamb.rhou
#rhoq = lamb.rhoq
#bemst = lamb.bemst
#emust = lamb.emust
#rho_accurate = lamb.rho_accurate


def retrieve_ssa(ssa_guess, pixel_index):
    # Skip the retrieval if the pixel quantities aren't good
    answer = np.zeros(n_wavelengths)
    if pixel_od[pixel_index] < 5 or solar_zenith_angle[pixel_index] > 72 or \
            emission_angle[pixel_index] > 72:
        answer[:] = np.nan
        print(f'skipping pixel {pixel_index}')
        return pixel_index, answer

    # Construct the EoS profiles
    # Use T profile from Kass et al 2019
    eos_file[:, 2] = np.array(
        [230, 230, 230, 230, 230, 230, 230, 230, 230, 230,
         230, 230, 230, 230, 230, 228, 226, 224, 222, 220,
         214, 208, 202, 196, 190, 186, 182, 178, 174, 170,
         166, 164, 158, 154, 150, 150, 150, 150, 150, 150,
         150])

    # Force P_surface to be 6.1 mbar
    eos_file[:, 1] *= eos_file[0, 1] / 610

    model_eos = eos_from_array(eos_file, z_boundaries, 3.71, 7.3 * 10 ** -26)
    temperatures = model_eos.temperature_boundaries
    h_lyr = model_eos.scale_height_boundaries

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Construct the mutable aerosol properties
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TODO: use real profile
    conrath_profile = Conrath(model_eos, 10, 0.01)
    # TODO: what is this really?
    p_sizes = np.linspace(1.5, 1.5, num=len(conrath_profile.profile))

    # Make Rayleigh stuff
    rco2 = RayleighCo2(short_wav[pixel_index, :], model_eos,
                       n_moments)

    def fit_ssa(guess, wav_index: int):
        # Trap the guess
        if not 0 <= guess <= 1:
            return 9999999

        # Take care of surface variables
        albedo = hapke[wav_index].albedo
        lamber = hapke[wav_index].lambertian
        rhou = hapke[wav_index].rhou
        rhoq = hapke[wav_index].rhoq
        bemst = hapke[wav_index].bemst
        emust = hapke[wav_index].emust
        rho_accurate = hapke[wav_index].rho_accurate

        cext = dust_file.array['primary'].data[:, :, 0]
        csca = dust_file.array['primary'].data[:, :, 1]
        csca[:, 0:2] = guess * cext[:, 0:2]
        c_ext = ForwardScatteringProperty(
            cext,
            particle_size_grid=sizes,
            wavelength_grid=wavs)
        c_sca = ForwardScatteringProperty(
            csca,
            particle_size_grid=sizes,
            wavelength_grid=wavs)

        dust_properties = ForwardScatteringPropertyCollection()
        dust_properties.add_property(c_ext, 'c_extinction')
        dust_properties.add_property(c_sca, 'c_scattering')

        # Use the Curiosity OD
        dust_col = Column(dust_properties, model_eos,
                          conrath_profile.profile, p_sizes,
                          short_wav[pixel_index, :], 0.88, pixel_od[pixel_index],
                          dust_phsfn)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Make the model
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model = ModelAtmosphere()
        # Make tuples of (dtauc, ssalb, pmom) for each constituent
        dust_info = (
        dust_col.total_optical_depth, dust_col.scattering_optical_depth,
        dust_col.scattering_optical_depth * dust_col.phase_function)
        rayleigh_info = (
        rco2.scattering_optical_depths, rco2.scattering_optical_depths,
        rco2.phase_function)  # This works since scat OD = total OD

        # Add dust and Rayleigh scattering to the model
        model.add_constituent(dust_info)
        model.add_constituent(rayleigh_info)

        model.compute_model()

        optical_depths = model.hyperspectral_total_optical_depths[:, wav_index]
        ssa = model.hyperspectral_total_single_scattering_albedos[:, wav_index]
        polynomial_moments = model.hyperspectral_legendre_moments[:, :, wav_index]

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
        utau = UserLevel(cp, mb).optical_depth_output

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Run the model
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
            disort.disort(usrang, usrtau, ibcnd, onlyfl, prnt, plank, lamber,
                          deltamplus, dopseudosphere, optical_depths, ssa,
                          polynomial_moments, temperatures, low_wavenumber,
                          high_wavenumber, utau,
                          mu0[pixel_index],
                          phi0[pixel_index],
                          mu[pixel_index],
                          phi[pixel_index], fbeam, fisot,
                          albedo, btemp, ttemp, temis, radius, h_lyr, rhoq, rhou,
                          rho_accurate, bemst, emust, accur, header, rfldir,
                          rfldn, flup, dfdt, uavg, uu, albmed, trnmed)
        return (uu[0, 0, 0] - reflectance[pixel_index, wav_index])**2
    for wavelength in range(n_wavelengths):
        fitted_ssa = optimize.minimize(fit_ssa, np.array([ssa_guess]), wavelength, method='Nelder-Mead').x
        answer[wavelength] = fitted_ssa
    return pixel_index, answer


def make_answer(inp):
    # inp must match the above: return pixel_index, answer
    retrieved_ssas[inp[0], :] = inp[1]

t0 = time.time()
n_cpus = mp.cpu_count()    # = 8 for my desktop, 12 for my laptop
pool = mp.Pool(7)   # save one just to be safe. Some say it's faster

for pixel in range(reflectance.shape[0]):
    pool.apply_async(retrieve_ssa, args=(0.65, pixel), callback=make_answer)
# https://www.machinelearningplus.com/python/parallel-processing-python/
pool.close()
pool.join()  # I guess this postpones further code execution until the queue is finished
np.save('/home/kyle/ssa_retrievals/retrieved_ssa_1-5_const_prop.npy', retrieved_ssas)
t1 = time.time()
print(t1-t0)
