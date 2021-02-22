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
from pyRT_DISORT.surface import Lambertian
from pyRT_DISORT.untested.aerosol import ForwardScatteringProperty, \
    ForwardScatteringPropertyCollection
from pyRT_DISORT.untested_utils.utilities.external_files import ExternalFile
from pyRT_DISORT.untested.aerosol_column import Column
from pyRT_DISORT.untested.vertical_profiles import Conrath
from pyRT_DISORT.untested.phase_function import TabularLegendreCoefficients
from pyRT_DISORT.untested.rayleigh import RayleighCo2
from pyRT_DISORT.untested.model_atmosphere import ModelAtmosphere

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in external aerosol files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in the atmosphere file
aero_path = '/home/kyle/repos/pyuvs-rt/data'
eos_file = ExternalFile(os.path.join(aero_path, 'marsatm.npy'))

# Read in the dust scattering properties file
dust_file = ExternalFile(os.path.join(aero_path, 'dust_properties.fits'))

# Read in the dust phase function file
dust_phsfn_file = ExternalFile(os.path.join(aero_path,
                                            'dust_phase_function.fits'))

# TODO: Read in external files (curiosity OD, vertical profiles) here and
#  interpolate such that I can get a value at any Ls

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the equation of state variables on a custom grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
z_boundaries = np.linspace(80, 0, num=20)    # Define the boundaries to use
model_eos = eos_from_array(eos_file.array, z_boundaries, 3.71, 7.3*10**-26)
temperatures = model_eos.temperature_boundaries
h_lyr = model_eos.scale_height_boundaries

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct the immutable aerosol properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
wavs = dust_file.array['wavelengths'].data
sizes = dust_file.array['particle_sizes'].data

# Make a phase function. I'm allowing negative coefficients here
dust_phsfn = TabularLegendreCoefficients(
    dust_phsfn_file.array['primary'].data,
    dust_phsfn_file.array['particle_sizes'].data,
    dust_phsfn_file.array['wavelengths'].data)

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
# TODO: This is just a test file. Select the data I want here
files = ['/home/kyle/Downloads/mvn_iuv_l1c_apoapse-orbit07287-muv_20180627T204223_v13_r01.fits']
for file in files:
    hdul = fits.open(file)
    reflectance = hdul['reflectance'].data
    n_integrations = reflectance.shape[0]
    n_positions = reflectance.shape[1]
    n_wavelengths = reflectance.shape[2]

    bin_width = 2.81   # nm
    wavelengths = hdul['wavelengths'].data

    short_wav = (wavelengths - bin_width) / 1000   # convert to microns
    long_wav = (wavelengths + bin_width) / 1000
    wavelengths = Wavelengths(short_wav, long_wav)
    low_wavenumber = wavelengths.low_wavenumber
    high_wavenumber = wavelengths.high_wavenumber

    sza = hdul['solar_zenith_angle'].data
    emission_angle = hdul['emission_angle'].data
    phase_angle = hdul['phase_angle'].data
    lt = hdul['local_time'].data
    
    angles = Angles(sza, emission_angle, phase_angle)
    mu = angles.mu
    mu0 = angles.mu0
    phi = angles.phi
    phi0 = angles.phi0

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Make a shared array
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    memmap_filename = os.path.join(mkdtemp(), 'myNewFile.dat')
    retrieved_ssas = np.memmap(memmap_filename, dtype=np.float, shape=reflectance.shape,
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
    lamb = Lambertian(0.01, cp)
    albedo = lamb.albedo
    lamber = lamb.lambertian
    rhou = lamb.rhou
    rhoq = lamb.rhoq
    bemst = lamb.bemst
    emust = lamb.emust
    rho_accurate = lamb.rho_accurate

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over pixel
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    t0 = time.time()
    for integration in range(n_integrations):
        for position in range(n_positions):
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Construct the mutable aerosol properties
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # TODO: use real profile
            conrath_profile = Conrath(model_eos, 10, 0.5)
            # TODO: what is this really?
            p_sizes = np.linspace(1.5, 1.5, num=len(conrath_profile.profile))

            # Make Rayleigh stuff
            rco2 = RayleighCo2(short_wav[integration, position, :], model_eos,
                               n_moments)

            answer = np.zeros(n_wavelengths)
            for wavelength in range(n_wavelengths):
                def fit_ssa(guess, wav):
                    # Trap the guess
                    if not 0 <= guess <= 1:
                        return 9999999
                    cext = dust_file.array['primary'].data[:, :, 0]
                    cext[:, 0:2] = 1  # TODO: are these indices correct?
                    csca = dust_file.array['primary'].data[:, :, 1]
                    csca[:, 0:2] = guess  # TODO: are these indices correct?
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

                    # Make the new Column where wave_ref = 9.3 microns and column OD = 7
                    # TODO: use real Curiosity OD
                    dust_col = Column(dust_properties, model_eos,
                                      conrath_profile.profile, p_sizes,
                                      short_wav[integration, position, :], 9.3, 7,
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

                    optical_depths = model.hyperspectral_total_optical_depths[:, wav]
                    ssa = model.hyperspectral_total_single_scattering_albedos[:, wav]
                    polynomial_moments = model.hyperspectral_legendre_moments[:, :, wav]

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
                                      mu0[integration, position],
                                      phi0[integration, position],
                                      mu[integration, position],
                                      phi[integration, position], fbeam, fisot,
                                      albedo, btemp, ttemp, temis, radius, h_lyr, rhoq, rhou,
                                      rho_accurate, bemst, emust, accur, header, rfldir,
                                      rfldn, flup, dfdt, uavg, uu, albmed, trnmed)
                    return (uu[0, 0, 0] - reflectance[integration, position, wav])**2
                fitted_ssa = optimize.minimize(fit_ssa, np.array([0.6]), wavelength, method='Nelder-Mead').x
                answer[wavelength] = fitted_ssa
            retrieved_ssas[position, integration, :] = answer
            t1 = time.time()
            print(t1 - t0)
            print(answer)
            raise SystemExit(9)

if __name__ == '__main__':
    n_cpus = mp.cpu_count()    # = 8 for my desktop, 12 for my laptop
    pool = mp.Pool(n_cpus - 1)   # save one just to be safe. Some say it's faster too





    # m = joblib.Parallel(n_jobs=-2, prefer='processes')(joblib.delayed(myModel)([0.8, 0.2], f) for f in pixel_inds)
    joblib.Parallel(n_jobs=-2, prefer='processes')(
        joblib.delayed(myModel)([0.8, 0.2], f) for f in pixel_inds)