import os, sys, pymysql, tempfile, sqlcl

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.io.fits as pyfits
from astropy.visualization.wcsaxes import add_beam, add_scalebar
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from regions.core import PixCoord
from regions.shapes.circle import CirclePixelRegion
from regions import RectanglePixelRegion, CircleAnnulusPixelRegion, EllipsePixelRegion

from itertools import repeat
from scipy.ndimage import rotate
from scipy.ndimage import binary_dilation

from spectral_cube import SpectralCube
from astropy.convolution import Gaussian1DKernel
import warnings
from spectral_cube.utils import SpectralCubeWarning
warnings.filterwarnings(action='ignore', category=SpectralCubeWarning, append=True)



def MH2_from_mom0(ID, z, ra, dec, r_outer, mask_type = 'total', a_CO = 4.35):

    # compute the molecular gas mass from PHANGS moment 0 maps, assuming a_CO = 4.35 (K km s−1 pc2)^−1 from Lin+20

    imagename = f'/arc/projects/salvage/ALMA_reduction/phangs_pipeline/derived/{ID}/{ID}_12m_co10_MOMENTTYPE.fits'

    moment_0     = pyfits.getdata(imagename.replace('MOMENTTYPE', 'strict_mom0'))
    moment_0_err = pyfits.getdata(imagename.replace('MOMENTTYPE', 'strict_emom0'))
    header_0     = pyfits.getheader(imagename.replace('MOMENTTYPE', 'strict_mom0'))

    # compute pc_per_pix from redshift and pixel scale
    arcsec_per_pix = header_0['CDELT2']*3600 # "/pix
    pc_per_arcsec = (cosmo.arcsec_per_kpc_proper(z=z).value / 1e3) ** -1
    pc_per_pix = pc_per_arcsec * arcsec_per_pix

    if mask_type == 'total':

        # don't need to do any prep.
        print()

    if mask_type == 'inner':

        # find central pixel from SDSS ra and dec
        wcs = WCS(header_0)
        center_coord = SkyCoord(ra, dec, unit="deg") 
        center_x, center_y = wcs.world_to_pixel(center_coord)

        # generate a circular aperture
        #center_pixel = header_0['CRPIX1']
        #center = PixCoord(center_pixel, center_pixel)
        center = PixCoord(center_x, center_y)
        radius = 1.5 / (header_0['CDELT2']*3600)
        aperture = CirclePixelRegion(center, radius)

        # conver aperture to a mask and apply to the moment map
        # mode = 'exact' uses partial contribution from edge pixels to simulate a perfect circle
        mask = aperture.to_mask(mode='exact')
        moment_0 = mask.multiply(moment_0)
        moment_0_err = mask.multiply(moment_0_err)

    if mask_type == 'outer':

        # instead of using the anulus pixel region, subtract the fancy circle mask from the outer circle
        # anulus method does not allow pixels to partially contribute to the inner and outer regions

        # both circular apertures will use the same center
        #center_pixel = header_0['CRPIX1']
        #center = PixCoord(center_pixel, center_pixel)
        wcs = WCS(header_0)
        center_coord = SkyCoord(ra, dec, unit="deg") 
        center_x, center_y = wcs.world_to_pixel(center_coord)
        center = PixCoord(center_x, center_y)

        # generate an inner circular mask with mode = 'exact'
        radius = 1.5 / (header_0['CDELT2']*3600)
        aperture = CirclePixelRegion(center, radius)
        mask = aperture.to_mask(mode='exact')

        # generate an outer circular mask with same center
        radius_outer = r_outer / (header_0['CDELT2']*3600)
        aperture = CirclePixelRegion(center, radius_outer)
        mask_outer = aperture.to_mask()

        # subtract the to masks as arrays
        mask_outer = np.array(mask_outer.to_image(np.shape(moment_0)).data, dtype = float) - np.array(mask.to_image(np.shape(moment_0)).data, dtype = float)

        moment_0 = mask_outer * moment_0
        moment_0_err = mask_outer * moment_0_err


    # sum moment map to get L_CO
    L_CO = np.nansum(np.nansum( moment_0 * pc_per_pix**2 ))
    L_CO_err = np.nansum(np.sqrt(np.nansum( (moment_0_err * pc_per_pix**2)**2 ) ))

    # convert to molecular gas mass
    M_H2 = L_CO * a_CO
    M_H2_err = L_CO_err * a_CO

    return M_H2, M_H2_err
    

def MH2_from_cube(ID, z, ra, dec, r_a, r_b, phi, a_CO = 4.35, visualize_spectra = True):

    '''
    Measures the molecular gas mass from ALMA CO(1-0) cube. 

    Input Parameters:
    ____________
    
    - ID:  SDSS objID of the target, used to import correct cube. [unitless]
    - z:   SDSS spectroscopic redshift, used to compute expected line velocity and pc/arcsec conversion. [unitless]
    - ra:  RA of SDSS fibre, used to extract mass from fibre location. [degrees]
    - dec: Dec. of the SDSS fibre, used to extract mass from fibre location. [degrees]
    - r_a: Semi-major axis of the model mag fit to SDSS r-band photmetry, used to generate total aperture. [arcsec]
    - r_b: Semi-minor axis of the model mag fit to SDSS r-band photmetry, used to generate total aperture. [arcsec]
    - phi: Orientation of the model mag fit to SDSS r-band photmetry, used to generate total aperture. [degrees]
    - a_CO: alpha_CO conversion factor, CO(1-0) flux to total molecular gas mass, default 4.35. [Msun pc−2 (K km s−1)−1]
    - visualize_spectra: option to skip plotting of the integrated spectra, default True.

    Returns:
    ____________
    
    - M_H2_inner: The measured molecular gas mass from central fibre aperture. [Msun]
    - M_H2_outer: The measured molecular gas mass from annulus aperture between SDSS fibre and SDSS ModelMag outer limit. [Msun]
    - M_H2_total: The measured molecular gas mass from central elliptical aperture defined by the SDSS ModelMag outer limit. [Msun]
    - M_H2_inner_err: The error on the inner molecular gas measurement. [Msun]
    - M_H2_outer_err: The error on the outer molecular gas measurement. [Msun]
    - M_H2_total_err: The error on the total molecular gas measurement. [Msun]
    '''

    file = f'/arc/projects/salvage/ALMA_reduction/phangs_pipeline/imaging/{ID}/{ID}_12m_co10.fits'
    header_image = pyfits.getheader(file)
    
    # import cube, convert to K
    cube = SpectralCube.read(file, format='fits').to(u.K).with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=(115.27120180 * u.GHz)/(1+z))
    spectral_axis = cube.spectral_axis

    # replace nans with 0's, don't want them adding to the integrated spectra...
    cube = cube.apply_numpy_function(np.nan_to_num, fill=0)
    
    # compute pc_per_pix from redshift and pixel scale
    arcsec_per_pix = header_image['CDELT2']*3600 # "/pix
    pc_per_arcsec = (cosmo.arcsec_per_kpc_proper(z=z).value / 1e3) ** -1
    pc_per_pix = pc_per_arcsec * arcsec_per_pix

    # find central pixel from SDSS ra and dec
    wcs = WCS(header_image)[0]
    center_coord = SkyCoord(ra, dec, unit="deg") 
    center_x, center_y = wcs.world_to_pixel(center_coord)
    center = PixCoord(center_x, center_y)

    # generate an inner circular aperture with radius 1.5"
    # mode = 'exact' uses partial contribution from edge pixels to simulate a perfect circle
    radius = 1.5 / (header_image['CDELT2']*3600)
    aperture = CirclePixelRegion(center, radius)
    mask_inner = aperture.to_mask(mode='exact')
    mask_inner = np.array(mask_inner.to_image(np.shape(cube[0,:,:])).data, dtype = float)
    
    # instead of using the anulus pixel region, subtract the fancy circle mask from the outer circle
    # anulus method does not allow pixels to partially contribute to the inner and outer regions
    
    # generate an outer elliptical aperture according to the SDSS modelMag photometry
    aperture = EllipsePixelRegion(center, r_a / (header_image['CDELT2']*3600), r_b / (header_image['CDELT2']*3600), phi * u.deg)
    mask_total = aperture.to_mask()
    mask_total = np.array(mask_total.to_image(np.shape(cube[0,:,:])).data, dtype = float)
    
    # annular outer mask
    mask_outer = mask_total - mask_inner
    mask_outer[mask_outer < 0] = 0 # to avoid over-subtraction if inner radius is larger than r_b
    
    # apply masks to cube
    masked_cube_inner = cube * mask_inner
    masked_cube_outer = cube * mask_outer
    masked_cube_total = cube * mask_total

    # integrate over the masked cubes' to extract spectra
    spec_inner = np.nansum(masked_cube_inner, axis = (1,2))
    spec_outer = np.nansum(masked_cube_outer, axis = (1,2))
    spec_total = np.nansum(masked_cube_total, axis = (1,2))

    # apply PHANGS cube-masking principles to identify voxels with signal along integrated spectra
    mask_signal_inner, rms_inner = identify_signal(spec_inner, spectral_axis, nchan_hi=3, snr_hi=1.0, nchan_lo=1, snr_lo=0.5, expand_by_nchan=1)
    mask_signal_outer, rms_outer = identify_signal(spec_outer, spectral_axis, nchan_hi=3, snr_hi=1.0, nchan_lo=1, snr_lo=0.5, expand_by_nchan=1)
    mask_signal_total, rms_total = identify_signal(spec_total, spectral_axis, nchan_hi=3, snr_hi=1.0, nchan_lo=1, snr_lo=0.5, expand_by_nchan=1)

    # velocity bin width
    dv = np.abs(header_image['CDELT3']/-1e3)

    # measure molecular gas masses based on the flux of identified CO signal
    M_H2_inner = np.nansum(spec_inner[mask_signal_inner]) * dv * pc_per_pix**2  * a_CO
    M_H2_outer = np.nansum(spec_outer[mask_signal_outer]) * dv * pc_per_pix**2  * a_CO
    M_H2_total = np.nansum(spec_total[mask_signal_total]) * dv * pc_per_pix**2  * a_CO

    # measure MH2 error via Equation (2) in Brown et al. (2021)
    M_H2_inner_err = rms_inner * dv * np.sqrt(len(spec_inner[mask_signal_inner])) * pc_per_pix**2 * a_CO
    M_H2_outer_err = rms_outer * dv * np.sqrt(len(spec_outer[mask_signal_outer])) * pc_per_pix**2 * a_CO
    M_H2_total_err = rms_total * dv * np.sqrt(len(spec_total[mask_signal_total])) * pc_per_pix**2 * a_CO

    ### visualize spectrum ###

    if visualize_spectra:
                                                                                                                                    
        fig, ax = plt.subplots(1,1,figsize = (8,5))
        
        ax.plot(spectral_axis, spec_inner, ds = 'steps-mid', label = 'Inner', color = 'orangered', alpha = 0.8, lw = 2)
        ax.plot(spectral_axis, spec_outer, ds = 'steps-mid', label = 'Outer', color = 'cornflowerblue', alpha = 0.8, lw = 2)
        #ax.plot(spectral_axis, spec_total, ds = 'steps-mid', label = 'Total', color = 'forestgreen', alpha = 0.8, lw = 2)
    
        ax.legend(fancybox = True, loc = 'upper left', frameon = False)
    
        ax.set_title('Integrated Spectra')
        ax.set_xlabel('Velocity [km/s]', fontsize = 11)
        ax.set_ylabel('Brightness Temperature [K]', fontsize = 11)
        
        ax.set_xticks(np.arange(-600,800,200))
        ax.set_xticks(np.arange(-600,800,100), minor = True)
        ax.set_yticks(np.arange(np.round(np.nanmin(spec_total), -1), np.round(np.nanmax(spec_total), -1)+10,10))
        ax.set_yticks(np.arange(np.round(np.nanmin(spec_total), -1), np.round(np.nanmax(spec_total), -1)+5,5), minor = True)
        
        ax.set_ylim(np.nanmin(spec_total) - 0.05 * np.nanmax(spec_total),np.nanmax(spec_total) + 0.05 * np.nanmax(spec_total))
        
        ax.tick_params('both', direction='in', which = 'both', top = True, right = True, width = 1., labelsize = 11)
        ax.tick_params(axis = 'both', which = 'major', length = 7)
        ax.tick_params(axis = 'both', which = 'minor', length = 4)
    
        ax.fill_betweenx([np.nanmin(spec_total) - 0.05 * np.nanmax(spec_total), np.nanmax(spec_total) + 0.05 * np.nanmax(spec_total)], np.nanmin(spectral_axis).value, np.nanmin(spectral_axis).value + 200, alpha = 0.2, hatch = '/', color = 'k')
        ax.fill_betweenx([np.nanmin(spec_total) - 0.05 * np.nanmax(spec_total), np.nanmax(spec_total) + 0.05 * np.nanmax(spec_total)], np.nanmax(spectral_axis).value - 200, np.nanmax(spectral_axis).value, alpha = 0.2, hatch = '/', color = 'k')
    
        spec_to_plot_inner = spec_inner.copy()
        spec_to_plot_inner[~mask_signal_inner] = 0
    
        spec_to_plot_outer = spec_outer.copy()
        spec_to_plot_outer[~mask_signal_outer] = 0
    
        ax.fill_between(spectral_axis, [0 for i in spec_inner], spec_to_plot_inner, step = 'mid', color = 'orangered', alpha = 0.3)
        ax.fill_between(spectral_axis, [0 for i in spec_outer], spec_to_plot_outer, step = 'mid', color = 'cornflowerblue', alpha = 0.3)  
        
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.)
        
        ax.set_title(f'Integrated Spectra')
    
        ax.text(300, np.nanmax(spec_total) - 0.1 * np.nanmax(spec_total), f'n-sigma (inner): {M_H2_inner/M_H2_inner_err:.2f}\nn-sigma (outer): {M_H2_outer/M_H2_outer_err:.2f}')
        
        plt.show()                                                                                                                               

    return M_H2_inner, M_H2_outer, M_H2_total, M_H2_inner_err, M_H2_outer_err, M_H2_total_err


def identify_signal(spec, vel, nchan_hi, snr_hi, nchan_lo, snr_lo, expand_by_nchan):

    '''
    Identifies signal along a spectrum following the methods used by VERTICO, PHANGS moment map methods.
    Inspired by identify_signal() in SpectralCubeTools by Jiayi Sun
    https://github.com/astrojysun/SpectralCubeTools/blob/master/spectral_cube_tools/identify_signal.py#L5

    Input Parameters:
    ____________
    
    - spec:     spectrum of interest. [arbitrary flux units]
    - vel:      spectal axis of spectrum. [km/s]
    - nchan_hi: number of contiguous channels that must meet the snr_hi criterion to constitute a core mask.
    - snr_hi:   signal-to-noise ratio of core mask voxels.
    - nchan_lo: number of contiguous channels that must meet the snr_lo criterion to constitute a wing mask.
    - snr_lo:   signal-to-noise ratio of wing mask voxels.

    Returns:
    _____________
    
    - mask_signal: Mask along spectral axis denoting which pixels should be included in flux measurements.
    - rms:         Root mean square error of spectrum, calculated from peripheral 200 km/s of spectrum. 
    '''

    # generate a region over which the flux will be measured
    rms_mask = (vel < (np.nanmin(vel) + (200 * u.km / u.s))) | (vel > (np.nanmax(vel) - (200 * u.km / u.s)))                                              
    rms = np.nanstd(spec[rms_mask])

    # mask of voxels I'm willing to include in the flux measurement
    mask = ~rms_mask & (spec>0)

    # compute SNR across the spectrum
    snr = spec / rms

    # initialize core mask
    mask_core = snr > snr_hi

    # enforce nchan_hi contiguous voxels
    for iiter in range(nchan_hi-1):
        mask_core &= np.roll(mask_core, 1, 0)
    
    for iiter in range(nchan_hi-1):
        mask_core |= np.roll(mask_core, -1, 0)

    mask_core &= mask
    
    # initialize wing mask
    mask_wing = snr > snr_lo

    # enforce nchan_lo contiguous voxels
    for iiter in range(nchan_lo-1):
        mask_wing &= np.roll(mask_wing, 1, 0)
    
    for iiter in range(nchan_lo-1):
        mask_wing |= np.roll(mask_wing, -1, 0)

    mask_wing &= mask

    # dilate core mask inside wing mask
    mask_signal = binary_dilation(mask_core, iterations=0, mask=mask_wing)

    # expand final mask by expand_by_nchan voxels (in each direction)
    if expand_by_nchan > 0:

        for iiter in range(expand_by_nchan):
            tempmask = np.roll(mask_signal, 1, axis=0)
            mask_signal |= tempmask
            tempmask = np.roll(mask_signal, -1, axis=0)
            mask_signal |= tempmask

    mask_signal &= mask

    return mask_signal, rms


def generate_masks(ra, dec, r_inner, ra_outer, rb_outer, phi, wcs, shape):

    '''

    Generate masks that correspond to the fibre aperture and SDSS ModelMag photometric limits. 

    Input Parameters:
    ____________
    
    - ra:    RA of SDSS fibre, used to extract mass from fibre location. [degrees]
    - dec:   Dec. of the SDSS fibre, used to extract mass from fibre location. [degrees]
    - r_a:   Semi-major axis of the model mag fit to SDSS r-band photmetry, used to generate total aperture. [cube pixels]
    - r_b:   Semi-minor axis of the model mag fit to SDSS r-band photmetry, used to generate total aperture. [cube pixels]
    - phi:   Orientation of the model mag fit to SDSS r-band photmetry, used to generate total aperture. [degrees]
    - wcs:   The world coordinate system object of the ALMA cube.
    - shape: The x,y shape of a slice through the cube along the spectral axis.

    Returns:
    ____________
    
    - mask_inner: The mask corresponding to the 1.5"-radius circular central fibre aperture.
    - mask_outer: The mask corresponding to the area outside the 1.5"-radius circular central fibre aperture but within the 4Re extent of the SDSS ModelMag photometric fit.
    - mask_total: The mask corresponding to the area within the 4Re extent of the SDSS ModelMag photometric fit.

    '''

    # find central pixel from SDSS ra and dec
    center_coord = SkyCoord(ra, dec, unit="deg") 
    center_x, center_y = wcs.world_to_pixel(center_coord)
    center = PixCoord(center_x, center_y)

    # generate an inner circular aperture with radius 1.5"
    # mode = 'exact' uses partial contribution from edge pixels to simulate a perfect circle
    aperture = CirclePixelRegion(center, r_inner)
    mask_inner = aperture.to_mask(mode='exact')
    mask_inner = np.array(mask_inner.to_image(shape).data, dtype = float)
    
    # instead of using the anulus pixel region, subtract the fancy circle mask from the outer circle
    # anulus method does not allow pixels to partially contribute to the inner and outer regions
    
    # generate an outer elliptical aperture according to the SDSS modelMag photometry
    aperture = EllipsePixelRegion(center, ra_outer, rb_outer, phi * u.deg)
    mask_total = aperture.to_mask()
    mask_total = np.array(mask_total.to_image(shape).data, dtype = float)
    
    # annular outer mask
    mask_outer = mask_total - mask_inner
    mask_outer[mask_outer < 0] = 0 # to avoid over-subtraction if inner radius is larger than r_b

    return mask_inner, mask_outer, mask_total


def ICO_to_LCO_K(ICO, arcsec_per_pix, D_L, z):

    '''
    Calculate CO luminosity, LCO, from the integrated flux, ICO, following Eq. (3) of Solomon & Bout (2005).

    Input Parameters:
    ____________

    - ICO:            The velocity-integrated flux, SCO * dv. [K km/s]
    - arcsec_per_pix: Pixel scale of the cube. ["/pix]
    - D_L:            Luminosity distance. [Mpc]
    - z:              SDSS spectroscopic redshift.

    Returns:
    ____________

    - LCO: The CO luminosity. [K km/s pc^2?]

    '''

    omega = arcsec_per_pix**2
    LCO = 23.5 * omega * D_L**2 * ICO * (1+z)**-3

    return LCO

def ICO_to_LCO_K_SW(ICO, pc_per_pix):

    '''
    Calculate CO luminosity, LCO, from the integrated flux, ICO, following Scott's intuition (for better or for worse!).

    Input Parameters:
    ____________

    - ICO:   The velocity-integrated flux, SCO * dv. [K km/s]
    - pc_per_pix: Observed frequency of the line, assumed to be at the expected velocity according to the SDSS redshift. [GHz]
    - z:     SDSS spectroscopic redshift.

    Returns:
    ____________

    - LCO: The CO luminosity. [K km/s pc^2?]

    '''

    LCO = ICO * pc_per_pix**2

    return LCO

def ICO_to_LCO_K_SW2(ICO, pc_per_pix, z):

    '''
    Calculate CO luminosity, LCO, from the integrated flux, ICO, following Scott's intuition (for better or for worse!).

    Input Parameters:
    ____________

    - ICO:   The velocity-integrated flux, SCO * dv. [K km/s]
    - pc_per_pix: Observed frequency of the line, assumed to be at the expected velocity according to the SDSS redshift. [GHz]
    - z:     SDSS spectroscopic redshift.

    Returns:
    ____________

    - LCO: The CO luminosity. [K km/s pc^2?]

    '''

    LCO = ICO * pc_per_pix**2 * (1+z)

    return LCO

def ICO_to_LCO_Jy(ICO, v_obs, D_L, z):

    '''
    Calculate CO luminosity, LCO, from the integrated flux, ICO, following Eq. (3) of Solomon & Bout (2005).

    Input Parameters:
    ____________

    - ICO:   The velocity-integrated flux, SCO * dv. [Jy km/s]
    - v_obs: Observed frequency of the line, assumed to be at the expected velocity according to the SDSS redshift. [GHz]
    - D_L:   Luminosity distance. [Mpc]
    - z:     SDSS spectroscopic redshift.

    Returns:
    ____________

    - LCO: The CO luminosity. [K km/s pc^2?]

    '''

    LCO = 3.25e7 * ICO * v_obs**-2 * D_L**2 * (1+z)**-3

    return LCO


def MH2_from_cube_FWZI(ID, z, ra, dec, r_a, r_b, phi, a_CO = 4.35, visualize_spectra = True):

    '''
    Measures the molecular gas mass from ALMA CO(1-0) cube using Toby's full-width at zero intensity (FWZI) function)

    Input Parameters:
    ____________
    
    - ID:  SDSS objID of the target, used to import correct cube. [unitless]
    - z:   SDSS spectroscopic redshift, used to compute expected line velocity and pc/arcsec conversion. [unitless]
    - ra:  RA of SDSS fibre, used to extract mass from fibre location. [degrees]
    - dec: Dec. of the SDSS fibre, used to extract mass from fibre location. [degrees]
    - r_a: Semi-major axis of the model mag fit to SDSS r-band photmetry, used to generate total aperture. [arcsec]
    - r_b: Semi-minor axis of the model mag fit to SDSS r-band photmetry, used to generate total aperture. [arcsec]
    - phi: Orientation of the model mag fit to SDSS r-band photmetry, used to generate total aperture. [degrees]
    - a_CO: alpha_CO conversion factor, CO(1-0) flux to total molecular gas mass, default 4.35. [Msun pc−2 (K km s−1)−1]
    - visualize_spectra: option to skip plotting of the integrated spectra, default True.

    Returns:
    ____________
    
    - M_H2_inner: The measured molecular gas mass from central fibre aperture. [Msun]
    - M_H2_outer: The measured molecular gas mass from annulus aperture between SDSS fibre and SDSS ModelMag outer limit. [Msun]
    - M_H2_total: The measured molecular gas mass from central elliptical aperture defined by the SDSS ModelMag outer limit. [Msun]
    - M_H2_inner_err: The error on the inner molecular gas measurement. [Msun]
    - M_H2_outer_err: The error on the outer molecular gas measurement. [Msun]
    - M_H2_total_err: The error on the total molecular gas measurement. [Msun]
    '''

    file = f'/arc/projects/salvage/ALMA_reduction/phangs_pipeline/imaging/{ID}/{ID}_12m_co10.fits'
    header_image = pyfits.getheader(file)
    
    # import cube, convert to K
    cube = SpectralCube.read(file, format='fits').to(u.K).with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=(115.27120180 * u.GHz)/(1+z))
    #cube = cube.downsample_axis(2, axis = 0)
    spectral_axis = cube.spectral_axis

    # replace nans with 0's, don't want them adding to the integrated spectra...
    cube = cube.apply_numpy_function(np.nan_to_num, fill=0)

    # velocity bin width
    print(header_image['CDELT3'])
    dv = header_image['CDELT3']/-1e3 #* 2 #(downsampled by 2x)
    
    # compute pc_per_pix from redshift and pixel scale
    arcsec_per_pix = header_image['CDELT2']*3600 # "/pix
    pc_per_arcsec = (cosmo.arcsec_per_kpc_proper(z=z).value / 1e3) ** -1
    pc_per_pix = pc_per_arcsec * arcsec_per_pix

    mask_inner, mask_outer, mask_total = generate_masks(ra, dec,\
                                                        1.5 / (header_image['CDELT2']*3600),\
                                                        r_a / (header_image['CDELT2']*3600),\
                                                        r_b / (header_image['CDELT2']*3600),\
                                                        phi,\
                                                        WCS(header_image)[0],\
                                                        np.shape(cube[0,:,:]))
    
    # apply masks to cube
    masked_cube_inner = cube * mask_inner
    masked_cube_outer = cube * mask_outer
    masked_cube_total = cube * mask_total

    # integrate over the masked cubes' to extract spectra
    spec_inner = np.nansum(masked_cube_inner, axis = (1,2))
    spec_outer = np.nansum(masked_cube_outer, axis = (1,2))
    spec_total = np.nansum(masked_cube_total, axis = (1,2))

    if dv<0:
        #dv must be positive for FWZI to work
        spectral_axis = spectral_axis[::-1].copy()
        spec_inner    = spec_inner[::-1].copy()
        spec_outer    = spec_outer[::-1].copy()
        spec_total    = spec_total[::-1].copy()
        dv = np.abs(dv)

    # compute RMS from outer regions of the spectra
    rms_mask = (spectral_axis < (np.nanmin(spectral_axis) + (200 * u.km / u.s))) | (spectral_axis > (np.nanmax(spectral_axis) - (200 * u.km / u.s))) 

    rms_inner = rms = np.nanstd(spec_inner[rms_mask])
    rms_outer = rms = np.nanstd(spec_outer[rms_mask])
    rms_total = rms = np.nanstd(spec_total[rms_mask])

    # apply VERTICO FWZI method
    ilo_inner, ihi_inner = idx_fwzi(spec_inner[~rms_mask])
    ilo_outer, ihi_outer = idx_fwzi_3peak(spec_outer[~rms_mask])
    ilo_total, ihi_total = idx_fwzi(spec_total[~rms_mask])

    mask_signal_inner = (spectral_axis > spectral_axis[~rms_mask][ilo_inner]) & (spectral_axis < spectral_axis[~rms_mask][ihi_inner])
    mask_signal_outer = (spectral_axis > spectral_axis[~rms_mask][ilo_outer]) & (spectral_axis < spectral_axis[~rms_mask][ihi_outer])
    mask_signal_total = (spectral_axis > spectral_axis[~rms_mask][ilo_total]) & (spectral_axis < spectral_axis[~rms_mask][ihi_total])

    # measure molecular gas masses based on the flux of identified CO signal
    ICO_inner = np.nansum(spec_inner[mask_signal_inner]) * dv # K km/s
    ICO_outer = np.nansum(spec_outer[mask_signal_outer]) * dv # K km/s
    ICO_total = np.nansum(spec_total[mask_signal_total]) * dv # K km/s

    v_obs = (115.27120180 * u.GHz)/(1+z)
    D_L = cosmo.luminosity_distance(z=z).to(u.Mpc).value
    
    L_CO_inner = ICO_to_LCO_K(ICO_inner, arcsec_per_pix, D_L, z)
    L_CO_outer = ICO_to_LCO_K(ICO_outer, arcsec_per_pix, D_L, z)
    L_CO_total = ICO_to_LCO_K(ICO_total, arcsec_per_pix, D_L, z)

    M_H2_inner = L_CO_inner * a_CO
    M_H2_outer = L_CO_outer * a_CO
    M_H2_total = L_CO_total * a_CO

    print(np.log10(M_H2_inner))

    L_CO_inner = ICO_to_LCO_K_SW(ICO_inner, pc_per_pix)
    L_CO_outer = ICO_to_LCO_K_SW(ICO_outer, pc_per_pix)
    L_CO_total = ICO_to_LCO_K_SW(ICO_total, pc_per_pix)

    M_H2_inner = L_CO_inner * a_CO
    M_H2_outer = L_CO_outer * a_CO
    M_H2_total = L_CO_total * a_CO

    print(np.log10(M_H2_inner))

    L_CO_inner = ICO_to_LCO_K_SW2(ICO_inner, pc_per_pix, z)
    L_CO_outer = ICO_to_LCO_K_SW2(ICO_outer, pc_per_pix, z)
    L_CO_total = ICO_to_LCO_K_SW2(ICO_total, pc_per_pix, z)

    M_H2_inner = L_CO_inner * a_CO
    M_H2_outer = L_CO_outer * a_CO
    M_H2_total = L_CO_total * a_CO

    print(np.log10(M_H2_inner))


    # measure MH2 error via Equation (2) in Brown et al. (2021)
    M_H2_inner_err = ICO_to_LCO_K_SW2(rms_inner * dv * np.sqrt(len(spec_inner[mask_signal_inner])), pc_per_pix, z) * a_CO
    M_H2_outer_err = ICO_to_LCO_K_SW2(rms_outer * dv * np.sqrt(len(spec_outer[mask_signal_outer])), pc_per_pix, z) * a_CO
    M_H2_total_err = ICO_to_LCO_K_SW2(rms_total * dv * np.sqrt(len(spec_total[mask_signal_total])), pc_per_pix, z) * a_CO

    # measure an upper limit for non-detections, assume line width of 300 km/s, ~12 channels
    M_H2_inner_uplim = 5 * ICO_to_LCO_K_SW2(rms_inner * dv * np.sqrt(300/dv), pc_per_pix, z) * a_CO
    M_H2_outer_uplim = 5 * ICO_to_LCO_K_SW2(rms_outer * dv * np.sqrt(300/dv), pc_per_pix, z) * a_CO
    M_H2_total_uplim = 5 * ICO_to_LCO_K_SW2(rms_total * dv * np.sqrt(300/dv), pc_per_pix, z) * a_CO

    ### visualize spectrum ###

    if visualize_spectra:
                                                                                                                                    
        fig, ax = plt.subplots(1,1,figsize = (8,5))
        
        ax.plot(spectral_axis, spec_inner, ds = 'steps-mid', label = 'Inner', color = 'orangered', alpha = 0.8, lw = 2)
        ax.plot(spectral_axis, spec_outer, ds = 'steps-mid', label = 'Outer', color = 'cornflowerblue', alpha = 0.8, lw = 2)
        #ax.plot(spectral_axis, spec_total, ds = 'steps-mid', label = 'Total', color = 'forestgreen', alpha = 0.8, lw = 2)
    
        ax.legend(fancybox = True, loc = 'upper left', frameon = False)
    
        ax.set_title('Integrated Spectra')
        ax.set_xlabel('Velocity [km/s]', fontsize = 11)
        ax.set_ylabel('Brightness Temperature [K]', fontsize = 11)
        
        ax.set_xticks(np.arange(-600,800,200))
        ax.set_xticks(np.arange(-600,800,100), minor = True)
        ax.set_yticks(np.arange(np.round(np.nanmin(spec_total), -1), np.round(np.nanmax(spec_total), -1)+10,10))
        ax.set_yticks(np.arange(np.round(np.nanmin(spec_total), -1), np.round(np.nanmax(spec_total), -1)+5,5), minor = True)
        
        ax.set_ylim(np.nanmin(spec_total) - 0.05 * np.nanmax(spec_total),np.nanmax(spec_total) + 0.05 * np.nanmax(spec_total))
        
        ax.tick_params('both', direction='in', which = 'both', top = True, right = True, width = 1., labelsize = 11)
        ax.tick_params(axis = 'both', which = 'major', length = 7)
        ax.tick_params(axis = 'both', which = 'minor', length = 4)
    
        ax.fill_betweenx([np.nanmin(spec_total) - 0.05 * np.nanmax(spec_total), np.nanmax(spec_total) + 0.05 * np.nanmax(spec_total)], np.nanmin(spectral_axis).value, np.nanmin(spectral_axis).value + 200, alpha = 0.2, hatch = '/', color = 'k')
        ax.fill_betweenx([np.nanmin(spec_total) - 0.05 * np.nanmax(spec_total), np.nanmax(spec_total) + 0.05 * np.nanmax(spec_total)], np.nanmax(spectral_axis).value - 200, np.nanmax(spectral_axis).value, alpha = 0.2, hatch = '/', color = 'k')
    
        spec_to_plot_inner = spec_inner.copy()
        spec_to_plot_inner[~mask_signal_inner] = 0
    
        spec_to_plot_outer = spec_outer.copy()
        spec_to_plot_outer[~mask_signal_outer] = 0
    
        ax.fill_between(spectral_axis, [0 for i in spec_inner], spec_to_plot_inner, step = 'mid', color = 'orangered', alpha = 0.3)
        ax.fill_between(spectral_axis, [0 for i in spec_outer], spec_to_plot_outer, step = 'mid', color = 'cornflowerblue', alpha = 0.3)  
        
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.)
        
        ax.set_title(f'Integrated Spectra')
    
        ax.text(300, np.nanmax(spec_total) - 0.1 * np.nanmax(spec_total), f'n-sigma (inner): {M_H2_inner/M_H2_inner_err:.2f}\nn-sigma (outer): {M_H2_outer/M_H2_outer_err:.2f}')
        
        plt.show()     

    print(spectral_axis[~rms_mask][ilo_inner], spectral_axis[~rms_mask][ihi_inner])
    
    return M_H2_inner, M_H2_outer, M_H2_total, M_H2_inner_err, M_H2_outer_err, M_H2_total_err, M_H2_inner_uplim, M_H2_outer_uplim, M_H2_total_uplim

def MH2_from_cube_FWZI_Jy(ID, z, ra, dec, r_a, r_b, phi, a_CO = 4.35, visualize_spectra = True):

    '''
    Measures the molecular gas mass from ALMA CO(1-0) cube using Toby's full-width at zero intensity (FWZI) function)

    Input Parameters:
    ____________
    
    - ID:  SDSS objID of the target, used to import correct cube. [unitless]
    - z:   SDSS spectroscopic redshift, used to compute expected line velocity and pc/arcsec conversion. [unitless]
    - ra:  RA of SDSS fibre, used to extract mass from fibre location. [degrees]
    - dec: Dec. of the SDSS fibre, used to extract mass from fibre location. [degrees]
    - r_a: Semi-major axis of the model mag fit to SDSS r-band photmetry, used to generate total aperture. [arcsec]
    - r_b: Semi-minor axis of the model mag fit to SDSS r-band photmetry, used to generate total aperture. [arcsec]
    - phi: Orientation of the model mag fit to SDSS r-band photmetry, used to generate total aperture. [degrees]
    - a_CO: alpha_CO conversion factor, CO(1-0) flux to total molecular gas mass, default 4.35. [Msun pc−2 (K km s−1)−1]
    - visualize_spectra: option to skip plotting of the integrated spectra, default True.

    Returns:
    ____________
    
    - M_H2_inner: The measured molecular gas mass from central fibre aperture. [Msun]
    - M_H2_outer: The measured molecular gas mass from annulus aperture between SDSS fibre and SDSS ModelMag outer limit. [Msun]
    - M_H2_total: The measured molecular gas mass from central elliptical aperture defined by the SDSS ModelMag outer limit. [Msun]
    - M_H2_inner_err: The error on the inner molecular gas measurement. [Msun]
    - M_H2_outer_err: The error on the outer molecular gas measurement. [Msun]
    - M_H2_total_err: The error on the total molecular gas measurement. [Msun]
    '''

    file = f'/arc/projects/salvage/ALMA_reduction/phangs_pipeline/imaging/{ID}/{ID}_12m_co10.fits'
    header_image = pyfits.getheader(file)
    
    # import cube, convert to K
    cube = SpectralCube.read(file, format='fits').with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=(115.27120180 * u.GHz)/(1+z))
    #cube = cube.downsample_axis(2, axis = 0)
    spectral_axis = cube.spectral_axis

    # replace nans with 0's, don't want them adding to the integrated spectra...
    cube = cube.apply_numpy_function(np.nan_to_num, fill=0)

    # velocity bin width
    print(header_image['CDELT3'])
    dv = header_image['CDELT3']/-1e3 #* 2 #(downsampled by 2x)
    
    # compute pc_per_pix from redshift and pixel scale
    arcsec_per_pix = header_image['CDELT2']*3600 # "/pix
    pc_per_arcsec = (cosmo.arcsec_per_kpc_proper(z=z).value / 1e3) ** -1
    pc_per_pix = pc_per_arcsec * arcsec_per_pix

    mask_inner, mask_outer, mask_total = generate_masks(ra, dec,\
                                                        1.5 / (header_image['CDELT2']*3600),\
                                                        r_a / (header_image['CDELT2']*3600),\
                                                        r_b / (header_image['CDELT2']*3600),\
                                                        phi,\
                                                        WCS(header_image)[0],\
                                                        np.shape(cube[0,:,:]))
    
    # apply masks to cube
    masked_cube_inner = cube * mask_inner
    masked_cube_outer = cube * mask_outer
    masked_cube_total = cube * mask_total

    # remove beam dependence
    beam_area = (np.pi / 4 / np.log(2)) * (header_image['BMAJ'] / header_image['CDELT2']) * (header_image['BMIN'] / header_image['CDELT2'])
    masked_cube_inner /= beam_area
    masked_cube_outer /= beam_area
    masked_cube_total /= beam_area

    # integrate over the masked cubes' to extract spectra
    spec_inner = np.nansum(masked_cube_inner, axis = (1,2))
    spec_outer = np.nansum(masked_cube_outer, axis = (1,2))
    spec_total = np.nansum(masked_cube_total, axis = (1,2))

    if dv<0:
        #dv must be positive for FWZI to work
        spectral_axis = spectral_axis[::-1].copy()
        spec_inner    = spec_inner[::-1].copy()
        spec_outer    = spec_outer[::-1].copy()
        spec_total    = spec_total[::-1].copy()
        dv = np.abs(dv)

    # compute RMS from outer regions of the spectra
    rms_mask = (spectral_axis < (np.nanmin(spectral_axis) + (200 * u.km / u.s))) | (spectral_axis > (np.nanmax(spectral_axis) - (200 * u.km / u.s))) 

    rms_inner = rms = np.nanstd(spec_inner[rms_mask])
    rms_outer = rms = np.nanstd(spec_outer[rms_mask])
    rms_total = rms = np.nanstd(spec_total[rms_mask])

    # apply VERTICO FWZI method
    ilo_inner, ihi_inner = idx_fwzi(spec_inner[~rms_mask])
    ilo_outer, ihi_outer = idx_fwzi_3peak(spec_outer[~rms_mask])
    ilo_total, ihi_total = idx_fwzi(spec_total[~rms_mask])

    mask_signal_inner = (spectral_axis > spectral_axis[~rms_mask][ilo_inner]) & (spectral_axis < spectral_axis[~rms_mask][ihi_inner])
    mask_signal_outer = (spectral_axis > spectral_axis[~rms_mask][ilo_outer]) & (spectral_axis < spectral_axis[~rms_mask][ihi_outer])
    mask_signal_total = (spectral_axis > spectral_axis[~rms_mask][ilo_total]) & (spectral_axis < spectral_axis[~rms_mask][ihi_total])

    # measure molecular gas masses based on the flux of identified CO signal
    ICO_inner = np.nansum(spec_inner[mask_signal_inner]) * dv # K km/s
    ICO_outer = np.nansum(spec_outer[mask_signal_outer]) * dv # K km/s
    ICO_total = np.nansum(spec_total[mask_signal_total]) * dv # K km/s

    v_obs = 115.27120180/(1+z)
    D_L = cosmo.luminosity_distance(z=z).to(u.Mpc).value

    L_CO_inner = ICO_to_LCO_Jy(ICO_inner, v_obs, D_L, z)
    L_CO_outer = ICO_to_LCO_Jy(ICO_outer, v_obs, D_L, z)
    L_CO_total = ICO_to_LCO_Jy(ICO_total, v_obs, D_L, z)

    M_H2_inner = L_CO_inner * a_CO
    M_H2_outer = L_CO_outer * a_CO
    M_H2_total = L_CO_total * a_CO

    print(np.log10(M_H2_inner))


    # measure MH2 error via Equation (2) in Brown et al. (2021)
    M_H2_inner_err = ICO_to_LCO_Jy(rms_inner * dv * np.sqrt(len(spec_inner[mask_signal_inner])), v_obs, D_L, z) * a_CO
    M_H2_outer_err = ICO_to_LCO_Jy(rms_outer * dv * np.sqrt(len(spec_outer[mask_signal_outer])), v_obs, D_L, z) * a_CO
    M_H2_total_err = ICO_to_LCO_Jy(rms_total * dv * np.sqrt(len(spec_total[mask_signal_total])), v_obs, D_L, z) * a_CO

    # measure an upper limit for non-detections, assume line width of 300 km/s, ~12 channels
    M_H2_inner_uplim = 5 * ICO_to_LCO_Jy(rms_inner * dv * np.sqrt(300/dv), v_obs, D_L, z) * a_CO
    M_H2_outer_uplim = 5 * ICO_to_LCO_Jy(rms_outer * dv * np.sqrt(300/dv), v_obs, D_L, z) * a_CO
    M_H2_total_uplim = 5 * ICO_to_LCO_Jy(rms_total * dv * np.sqrt(300/dv), v_obs, D_L, z) * a_CO

    ### visualize spectrum ###

    if visualize_spectra:
                                                                                                                                    
        fig, ax = plt.subplots(1,1,figsize = (8,5))
        
        ax.plot(spectral_axis, spec_inner, ds = 'steps-mid', label = 'Inner', color = 'orangered', alpha = 0.8, lw = 2)
        ax.plot(spectral_axis, spec_outer, ds = 'steps-mid', label = 'Outer', color = 'cornflowerblue', alpha = 0.8, lw = 2)
        #ax.plot(spectral_axis, spec_total, ds = 'steps-mid', label = 'Total', color = 'forestgreen', alpha = 0.8, lw = 2)
    
        ax.legend(fancybox = True, loc = 'upper left', frameon = False)
    
        ax.set_title('Integrated Spectra')
        ax.set_xlabel('Velocity [km/s]', fontsize = 11)
        ax.set_ylabel('Flux [Jy]', fontsize = 11)
        
        ax.set_xticks(np.arange(-600,800,200))
        ax.set_xticks(np.arange(-600,800,100), minor = True)
        ax.set_yticks(np.arange(np.round(np.nanmin(spec_total), -1), np.round(np.nanmax(spec_total), -1)+10,10))
        ax.set_yticks(np.arange(np.round(np.nanmin(spec_total), -1), np.round(np.nanmax(spec_total), -1)+5,5), minor = True)
        
        ax.set_ylim(np.nanmin(spec_total) - 0.05 * np.nanmax(spec_total),np.nanmax(spec_total) + 0.05 * np.nanmax(spec_total))
        
        ax.tick_params('both', direction='in', which = 'both', top = True, right = True, width = 1., labelsize = 11)
        ax.tick_params(axis = 'both', which = 'major', length = 7)
        ax.tick_params(axis = 'both', which = 'minor', length = 4)
    
        ax.fill_betweenx([np.nanmin(spec_total) - 0.05 * np.nanmax(spec_total), np.nanmax(spec_total) + 0.05 * np.nanmax(spec_total)], np.nanmin(spectral_axis).value, np.nanmin(spectral_axis).value + 200, alpha = 0.2, hatch = '/', color = 'k')
        ax.fill_betweenx([np.nanmin(spec_total) - 0.05 * np.nanmax(spec_total), np.nanmax(spec_total) + 0.05 * np.nanmax(spec_total)], np.nanmax(spectral_axis).value - 200, np.nanmax(spectral_axis).value, alpha = 0.2, hatch = '/', color = 'k')
    
        spec_to_plot_inner = spec_inner.copy()
        spec_to_plot_inner[~mask_signal_inner] = 0
    
        spec_to_plot_outer = spec_outer.copy()
        spec_to_plot_outer[~mask_signal_outer] = 0
    
        ax.fill_between(spectral_axis, [0 for i in spec_inner], spec_to_plot_inner, step = 'mid', color = 'orangered', alpha = 0.3)
        ax.fill_between(spectral_axis, [0 for i in spec_outer], spec_to_plot_outer, step = 'mid', color = 'cornflowerblue', alpha = 0.3)  
        
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.)
        
        ax.set_title(f'Integrated Spectra')
    
        ax.text(300, np.nanmax(spec_total) - 0.1 * np.nanmax(spec_total), f'n-sigma (inner): {M_H2_inner/M_H2_inner_err:.2f}\nn-sigma (outer): {M_H2_outer/M_H2_outer_err:.2f}')
        
        plt.show()     

    print(spectral_axis[~rms_mask][ilo_inner], spectral_axis[~rms_mask][ihi_inner])
    
    return M_H2_inner, M_H2_outer, M_H2_total, M_H2_inner_err, M_H2_outer_err, M_H2_total_err, M_H2_inner_uplim, M_H2_outer_uplim, M_H2_total_uplim

    
def idx_fwzi(data):
    """
    Compute indices at the true full width at zero intensity (i.e. the continuum level)
    of the spectrum.

    This makes no assumptions about the shape of the spectrum. It determines the
    index of the highest flux value, and then calculates indices at that base of
    the feature where intensity drops below 0.

    Parameters
    ----------
    data :
        The flux array over which the width will be calculated."""

    # Find the index of the maximum peak value
    peak_ind = list(np.where(data == np.nanmax(data))[0])

    # split the result into a high and low spectrum
    split = np.split(data, peak_ind)

    if np.nanmin(split[0]) > 0:
        ihi = 0
    else:
        ihi = (split[0].shape - np.argmax(split[0][::-1] < 0) - 1)[0]

    if np.nanmin(split[1]) > 0:
        ilo = len(data)-1
    else:
        #  len of lower portion (split point idx) plus the last point at which the upper half of the spec is less than 0
        ilo = (split[0].shape + np.argmax(split[1] < 0) - 1)[0] + 1

    return ilo, ihi


def idx_fwzi_3peak(data):
    """
    Compute indices at the true full width at zero intensity (i.e. the continuum level)
    of the spectrum.

    This makes no assumptions about the shape of the spectrum. It determines the
    index of the highest flux value, and then calculates indices at that base of
    the feature where intensity drops below 0.

    Parameters
    ----------
    data :
        The flux array over which the width will be calculated."""

    # Find the index of the maximum peak value
    peak_ind1 = list(np.where(data == np.nanmax(data))[0])

    # split the result into a high and low spectrum
    split = np.split(data, peak_ind1)
    
    if np.nanmin(split[0]) > 0:
        ihi1 = 0
    else:
        ihi1 = (split[0].shape - np.argmax(split[0][::-1] < 0) - 1)[0]

    if np.nanmin(split[1]) > 0:
        ilo1 = len(data)-1
    else:
        #  len of lower portion (split point idx) plus the last point at which the upper half of the spec is less than 0
        ilo1 = (split[0].shape + np.argmax(split[1] < 0) - 1)[0] + 1


    # Find the index of the second maximum peak value
    peak_ind2 = list(np.where(data == np.nanmax(data[data!=data[peak_ind1]]))[0])

    # split the result into a high and low spectrum
    split = np.split(data, peak_ind2)
    
    if np.nanmin(split[0]) > 0:
        ihi2 = 0
    else:
        ihi2 = (split[0].shape - np.argmax(split[0][::-1] < 0) - 1)[0]

    if np.nanmin(split[1]) > 0:
        ilo2 = len(data)-1
    else:
        #  len of lower portion (split point idx) plus the last point at which the upper half of the spec is less than 0
        ilo2 = (split[0].shape + np.argmax(split[1] < 0) - 1)[0] + 1

     # Find the index of the third maximum peak value
    peak_ind3 = list(np.where(data == np.nanmax(data[(data!=data[peak_ind1]) & (data!=data[peak_ind2])]))[0])

    # split the result into a high and low spectrum
    split = np.split(data, peak_ind3)
    
    if np.nanmin(split[0]) > 0:
        ihi3 = 0
    else:
        ihi3 = (split[0].shape - np.argmax(split[0][::-1] < 0) - 1)[0]

    if np.nanmin(split[1]) > 0:
        ilo3 = len(data)-1
    else:
        #  len of lower portion (split point idx) plus the last point at which the upper half of the spec is less than 0
        ilo3 = (split[0].shape + np.argmax(split[1] < 0) - 1)[0] + 1

    ilo = np.max([ilo1,ilo2,ilo3])
    ihi = np.min([ihi1,ihi2,ihi3])

    return ilo, ihi


def MH2_from_cube_iter(ID, z, ra, dec, r_a, r_b, phi, a_CO = 4.35):

    file = f'/arc/projects/salvage/ALMA_reduction/phangs_pipeline/imaging/{ID}/{ID}_12m_co10.fits'
    header_image = pyfits.getheader(file)
    
    # import cube, convert to K
    cube = SpectralCube.read(file, format='fits').to(u.K).with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=(115.27120180 * u.GHz)/(1+z))
    spectral_axis = cube.spectral_axis

    # replace nans with 0's, don't want them adding to the integrated spectra...
    cube = cube.apply_numpy_function(np.nan_to_num, fill=0)
    
    # compute pc_per_pix from redshift and pixel scale
    arcsec_per_pix = header_image['CDELT2']*3600 # "/pix
    pc_per_arcsec = (cosmo.arcsec_per_kpc_proper(z=z).value / 1e3) ** -1
    pc_per_pix = pc_per_arcsec * arcsec_per_pix

    # find central pixel from SDSS ra and dec
    wcs = WCS(header_image)[0]
    center_coord = SkyCoord(ra, dec, unit="deg") 
    center_x, center_y = wcs.world_to_pixel(center_coord)
    center = PixCoord(center_x, center_y)

    # generate an inner circular aperture with radius 1.5"
    # mode = 'exact' uses partial contribution from edge pixels to simulate a perfect circle
    radius = 1.5 / (header_image['CDELT2']*3600)
    aperture = CirclePixelRegion(center, radius)
    mask_inner = aperture.to_mask(mode='exact')
    mask_inner = np.array(mask_inner.to_image(np.shape(cube[0,:,:])).data, dtype = float)
    
    # instead of using the anulus pixel region, subtract the fancy circle mask from the outer circle
    # anulus method does not allow pixels to partially contribute to the inner and outer regions
    
    # generate an outer elliptical aperture according to the SDSS modelMag photometry
    aperture = EllipsePixelRegion(center, r_a / (header_image['CDELT2']*3600), r_b / (header_image['CDELT2']*3600), phi * u.deg)
    mask_total = aperture.to_mask()
    mask_total = np.array(mask_total.to_image(np.shape(cube[0,:,:])).data, dtype = float)
    
    # annular outer mask
    mask_outer = mask_total - mask_inner
    mask_outer[mask_outer < 0] = 0 # to avoid over-subtraction if inner radius is larger than r_b
    
    # apply masks to cube
    masked_cube_inner = cube * mask_inner
    masked_cube_outer = cube * mask_outer
    masked_cube_total = cube * mask_total

    # integrate over the masked cubes' to extract spectra
    spec_inner = np.nansum(masked_cube_inner, axis = (1,2))
    spec_outer = np.nansum(masked_cube_outer, axis = (1,2))
    spec_total = np.nansum(masked_cube_total, axis = (1,2))
    
    # bin for better sensitivity?
    #downsampled_stack = stack.downsample_axis(2, axis = 0)
    #downsampled_stack.quicklook(label = 'Total Integrated Spectrum (binned by 2x)')
    
    # smooth for visualization
    #kernel = Gaussian1DKernel(2)
    #smoothstack = stack.spectral_smooth(kernel)
    #smoothstack.quicklook(label = 'Total Integrated Spectrum (smoothed by 2x)')

    
    ### compute masses in iteratively expanding frequency window until S/N decreases ###
    ##  error on integrated intensity, following Brown et al. (2021), Equation (2)    ##
 
    M_H2_inner, M_H2_inner_err, SNR_inner, ngrows_inner = iterative_mass_measure(spec_inner, spectral_axis, np.abs(header_image['CDELT3']/-1e3), pc_per_pix, a_CO, 5) 
    M_H2_outer, M_H2_outer_err, SNR_outer, ngrows_outer = iterative_mass_measure(spec_outer, spectral_axis, np.abs(header_image['CDELT3']/-1e3), pc_per_pix, a_CO, 5) 
    M_H2_total, M_H2_total_err, SNR_total, ngrows_total = iterative_mass_measure(spec_total, spectral_axis, np.abs(header_image['CDELT3']/-1e3), pc_per_pix, a_CO, 5) 

    if M_H2_inner < 0: M_H2_inner = 0
    if M_H2_outer < 0: M_H2_outer = 0
    if M_H2_total < 0: M_H2_total = 0

    ### visualize spectrum ###
                                                                                                                                    
    fig, ax = plt.subplots(1,1,figsize = (8,5))
    
    ax.plot(spectral_axis, spec_inner, ds = 'steps-mid', label = 'Inner', color = 'orangered', alpha = 0.8, lw = 2)
    ax.plot(spectral_axis, spec_outer, ds = 'steps-mid', label = 'Outer', color = 'cornflowerblue', alpha = 0.8, lw = 2)
    #ax.plot(spectral_axis, spec_total, ds = 'steps-mid', label = 'Total', color = 'forestgreen', alpha = 0.8, lw = 2)

    ax.legend(fancybox = True, loc = 'upper left', frameon = False)

    ax.set_title('Integrated Spectra')
    ax.set_xlabel('Velocity [km/s]', fontsize = 11)
    ax.set_ylabel('Brightness Temperature [K]', fontsize = 11)
    
    #ax.set_xticks(np.arange(np.min(spec_axis).value,np.max(spec_axis).value+200,200))
    #ax.set_xticks(np.arange(np.min(spec_axis).value,np.max(spec_axis).value+100,100), minor = True)
    ax.set_xticks(np.arange(-600,800,200))
    ax.set_xticks(np.arange(-600,800,100), minor = True)
    ax.set_yticks(np.arange(np.round(np.nanmin(spec_total), -1), np.round(np.nanmax(spec_total), -1)+10,10))
    ax.set_yticks(np.arange(np.round(np.nanmin(spec_total), -1), np.round(np.nanmax(spec_total), -1)+5,5), minor = True)
    
    #ax.set_xlim(np.min(spec_axis).value,np.max(spec_axis).value)
    #ax.set_xlim(-650, 650)
    ax.set_ylim(np.nanmin(spec_total) - 0.05 * np.nanmax(spec_total),np.nanmax(spec_total) + 0.05 * np.nanmax(spec_total))
    
    ax.tick_params('both', direction='in', which = 'both', top = True, right = True, width = 1., labelsize = 11)
    ax.tick_params(axis = 'both', which = 'major', length = 7)
    ax.tick_params(axis = 'both', which = 'minor', length = 4)

    ax.fill_betweenx([np.nanmin(spec_total) - 0.05 * np.nanmax(spec_total), np.nanmax(spec_total) + 0.05 * np.nanmax(spec_total)], np.nanmin(spectral_axis).value, -ngrows_outer * 50, alpha = 0.2, hatch = '/', color = 'k')
    ax.fill_betweenx([np.nanmin(spec_total) - 0.05 * np.nanmax(spec_total), np.nanmax(spec_total) + 0.05 * np.nanmax(spec_total)], ngrows_outer * 50, np.nanmax(spectral_axis).value, alpha = 0.2, hatch = '/', color = 'k')

    ax.axvline(-ngrows_inner * 50, color = 'tab:orange', ls = '--', alpha = 0.5)
    ax.axvline(ngrows_inner * 50, color = 'tab:orange', ls = '--', alpha = 0.5)

    #ax.fill_between([10, 20], np.nanmin(spectral_axis).value, (np.nanmin(spectral_axis).value + 250), alpha = 0.4, hatch = '/', color = 'k')
    #ax.fill_between([0,100], np.nanmax(spectral_axis).value - 100, np.nanmax(spectral_axis).value, alpha = 0.4, hatch = '/', color = 'k')
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.)
    
    ax.set_title(f'Integrated Spectra')

    ax.text(300, np.nanmax(spec_total) - 0.1 * np.nanmax(spec_total), f'n-sigma (inner): {M_H2_inner/M_H2_inner_err:.2f}\nn-sigma (outer): {M_H2_outer/M_H2_outer_err:.2f}')
    
    plt.show()                                                                                                                               

    return M_H2_inner, M_H2_outer, M_H2_total, M_H2_inner_err, M_H2_outer_err, M_H2_total_err

def iterative_mass_measure(spec, vel, dv, pc_per_pix, a_CO, N_grows):

    # default dummy SNR
    SNR = 0

    for n in np.arange(1,N_grows+1,1):

        print(n)

        # generate a window over which the flux will be measured
        spectral_mask = (vel < (n * 50 * u.km / u.s)) & (vel > (-n * 50 * u.km / u.s))

        # sum the flux and convert to mass
        M_H2_tmp     = np.nansum(spec[spectral_mask]) * pc_per_pix**2 * dv * a_CO

        # compute the error on the mass measurement following Brown et al. (2021), Equation (2)
        # compute over all channels not included in the flux measurement
        M_H2_err_tmp = np.nanstd(spec[~spectral_mask]) * pc_per_pix**2 * dv * np.sqrt(len(spec[spectral_mask])) * a_CO

        # measure the S/N
        SNR_tmp = M_H2_inner_tmp / M_H2_inner_err_tmp

        # if SNR is improved, make temporary values the new defacto values
        if SNR_tmp > SNR:            
            M_H2     = M_H2_tmp
            M_H2_err = M_H2_err_tmp
            SNR      = M_H2 / M_H2_err
            n_final  = n
        
        else:
            continue

        return M_H2, M_H2_err, SNR, n_final
    
def MH2_from_cube_phys(ID, z, ra, dec, r_inner, r_outer, a_CO = 4.35):

    file = f'/arc/projects/salvage/ALMA_reduction/phangs_pipeline/imaging/{ID}/{ID}_12m_co10.fits'
    header_image = pyfits.getheader(file)
    
    # import cube, convert to K
    cube = SpectralCube.read(file, format='fits').to(u.K).with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=(115.27120180 * u.GHz)/(1+z))
    spectral_axis = cube.spectral_axis

    # replace nans with 0's, don't want them adding to the integrated spectra...
    cube = cube.apply_numpy_function(np.nan_to_num, fill=0)
    
    # compute pc_per_pix from redshift and pixel scale
    arcsec_per_pix = header_image['CDELT2']*3600 # "/pix
    pc_per_arcsec = (cosmo.arcsec_per_kpc_proper(z=z).value / 1e3) ** -1
    pc_per_pix = pc_per_arcsec * arcsec_per_pix

    # find central pixel from SDSS ra and dec
    wcs = WCS(header_image)[0]
    center_coord = SkyCoord(ra, dec, unit="deg") 
    center_x, center_y = wcs.world_to_pixel(center_coord)
    center = PixCoord(center_x, center_y)

    # generate an inner circular aperture with radius 1.5"
    # mode = 'exact' uses partial contribution from edge pixels to simulate a perfect circle
    radius = r_inner / pc_per_pix
    aperture = CirclePixelRegion(center, radius)
    mask_inner = aperture.to_mask(mode='exact')
    mask_inner = np.array(mask_inner.to_image(np.shape(cube[0,:,:])).data, dtype = float)
        
    # instead of using the anulus pixel region, subtract the fancy circle mask from the outer circle
    # anulus method does not allow pixels to partially contribute to the inner and outer regions
    
    # generate an outer elliptical aperture according to the SDSS modelMag photometry
    radius_outer = r_outer / pc_per_pix
    aperture = CirclePixelRegion(center, radius_outer)
    mask_total = aperture.to_mask()
    mask_total = np.array(mask_total.to_image(np.shape(cube[0,:,:])).data, dtype = float)
    
    # annular outer mask
    mask_outer = mask_total - mask_inner
    mask_outer[mask_outer < 0] = 0 # to avoid over-subtraction if inner radius is larger than r_b
    
    # apply masks to cube
    masked_cube_inner = cube * mask_inner
    masked_cube_outer = cube * mask_outer
    masked_cube_total = cube * mask_total

    # integrate over the masked cubes' to extract spectra
    spec_inner = np.nansum(masked_cube_inner, axis = (1,2))
    spec_outer = np.nansum(masked_cube_outer, axis = (1,2))
    spec_total = np.nansum(masked_cube_total, axis = (1,2))
    
    ### compute masses ###
    
    spectral_mask = (spectral_axis < (np.nanmax(spectral_axis) - (200 * u.km / u.s))) & (spectral_axis > (np.nanmin(spectral_axis) + (200 * u.km / u.s)))
    
    L_CO_inner = np.nansum(spec_inner[spectral_mask]) * pc_per_pix**2 * np.abs(header_image['CDELT3']/-1e3)
    L_CO_outer = np.nansum(spec_outer[spectral_mask]) * pc_per_pix**2 * np.abs(header_image['CDELT3']/-1e3)
    L_CO_total = np.nansum(spec_total[spectral_mask]) * pc_per_pix**2 * np.abs(header_image['CDELT3']/-1e3)
    
    M_H2_inner = L_CO_inner * a_CO
    M_H2_outer = L_CO_outer * a_CO
    M_H2_total = L_CO_total * a_CO

    # error on integrated intensity, following Brown et al. (2021), Equation (2)
    M_H2_inner_err = (np.nanstd(spec_inner[~spectral_mask]) * pc_per_pix**2 * np.abs(header_image['CDELT3']/-1e3)) * a_CO * np.sqrt(len(spec_inner[spectral_mask])) #* np.abs(header_image['CDELT3'])/1e3
    M_H2_outer_err = (np.nanstd(spec_outer[~spectral_mask]) * pc_per_pix**2 * np.abs(header_image['CDELT3']/-1e3)) * a_CO * np.sqrt(len(spec_inner[spectral_mask])) #* np.abs(header_image['CDELT3'])/1e3
    M_H2_total_err = (np.nanstd(spec_total[~spectral_mask]) * pc_per_pix**2 * np.abs(header_image['CDELT3']/-1e3)) * a_CO * np.sqrt(len(spec_inner[spectral_mask])) #* np.abs(header_image['CDELT3'])/1e3

    if M_H2_inner < 0: M_H2_inner = 0
    if M_H2_outer < 0: M_H2_outer = 0
    if M_H2_total < 0: M_H2_total = 0

    ### visualize spectrum ###
                                                                                                                                    
    fig, ax = plt.subplots(1,1,figsize = (8,5))
    
    ax.plot(spectral_axis, spec_inner, ds = 'steps-mid', label = 'Inner', color = 'orangered', alpha = 0.8, lw = 2)
    ax.plot(spectral_axis, spec_outer, ds = 'steps-mid', label = 'Outer', color = 'cornflowerblue', alpha = 0.8, lw = 2)
    #ax.plot(spectral_axis, spec_total, ds = 'steps-mid', label = 'Total', color = 'forestgreen', alpha = 0.8, lw = 2)

    ax.legend(fancybox = True, loc = 'upper left', frameon = False)

    ax.set_title('Integrated Spectra')
    ax.set_xlabel('Velocity [km/s]', fontsize = 11)
    ax.set_ylabel('Brightness Temperature [K]', fontsize = 11)
    
    #ax.set_xticks(np.arange(np.min(spec_axis).value,np.max(spec_axis).value+200,200))
    #ax.set_xticks(np.arange(np.min(spec_axis).value,np.max(spec_axis).value+100,100), minor = True)
    ax.set_xticks(np.arange(-600,800,200))
    ax.set_xticks(np.arange(-600,800,100), minor = True)
    ax.set_yticks(np.arange(np.round(np.nanmin(spec_total), -1), np.round(np.nanmax(spec_total), -1)+10,10))
    ax.set_yticks(np.arange(np.round(np.nanmin(spec_total), -1), np.round(np.nanmax(spec_total), -1)+5,5), minor = True)
    
    #ax.set_xlim(np.min(spec_axis).value,np.max(spec_axis).value)
    #ax.set_xlim(-650, 650)
    ax.set_ylim(np.nanmin(spec_total) - 0.05 * np.nanmax(spec_total),np.nanmax(spec_total) + 0.05 * np.nanmax(spec_total))
    
    ax.tick_params('both', direction='in', which = 'both', top = True, right = True, width = 1., labelsize = 11)
    ax.tick_params(axis = 'both', which = 'major', length = 7)
    ax.tick_params(axis = 'both', which = 'minor', length = 4)

    ax.fill_betweenx([np.nanmin(spec_total) - 0.05 * np.nanmax(spec_total), np.nanmax(spec_total) + 0.05 * np.nanmax(spec_total)], np.nanmin(spectral_axis).value, (np.nanmin(spectral_axis).value + 200), alpha = 0.2, hatch = '/', color = 'k')
    ax.fill_betweenx([np.nanmin(spec_total) - 0.05 * np.nanmax(spec_total), np.nanmax(spec_total) + 0.05 * np.nanmax(spec_total)], (np.nanmax(spectral_axis).value - 200), np.nanmax(spectral_axis).value, alpha = 0.2, hatch = '/', color = 'k')

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.)
    
    ax.set_title(f'Integrated Spectra')

    ax.text(300, np.nanmax(spec_total) - 0.1 * np.nanmax(spec_total), f'n-sigma (inner): {M_H2_inner/M_H2_inner_err:.2f}\nn-sigma (outer): {M_H2_outer/M_H2_outer_err:.2f}')
    
    plt.show()                                                                                                                               

    return M_H2_inner, M_H2_outer, M_H2_total, M_H2_inner_err, M_H2_outer_err, M_H2_total_err


def mom0_pixel_coverage(ID, z, ra, dec, r_a, r_b, phi):

    # compute the molecular gas mass from PHANGS moment 0 maps, assuming a_CO = 4.35 (K km s−1 pc2)^−1 from Lin+20

    imagename = f'/arc/projects/salvage/ALMA_reduction/phangs_pipeline/derived/{ID}/{ID}_12m_co10_MOMENTTYPE.fits'

    moment_0 = pyfits.getdata(imagename.replace('MOMENTTYPE', 'strict_mom0'))
    header_0 = pyfits.getheader(imagename.replace('MOMENTTYPE', 'strict_mom0'))

    # compute pc_per_pix from redshift and pixel scale
    arcsec_per_pix = header_0['CDELT2']*3600 # "/pix
    pc_per_arcsec = (cosmo.arcsec_per_kpc_proper(z=z).value / 1e3) ** -1
    pc_per_pix = pc_per_arcsec * arcsec_per_pix

    # find central pixel from SDSS ra and dec
    wcs = WCS(header_0)
    center_coord = SkyCoord(ra, dec, unit="deg") 
    center_x, center_y = wcs.world_to_pixel(center_coord)
    center = PixCoord(center_x, center_y)

    # generate an inner circular aperture with radius 1.5"
    # mode = 'exact' uses partial contribution from edge pixels to simulate a perfect circle
    radius = 1.5 / (header_0['CDELT2']*3600)
    aperture = CirclePixelRegion(center, radius)
    mask_inner = aperture.to_mask(mode='exact')
    mask_inner = np.array(mask_inner.to_image(np.shape(moment_0)), dtype = float)
    
    # instead of using the anulus pixel region, subtract the fancy circle mask from the outer circle
    # anulus method does not allow pixels to partially contribute to the inner and outer regions
    
    # generate an outer elliptical aperture according to the SDSS modelMag photometry
    aperture = EllipsePixelRegion(center, r_a / (header_0['CDELT2']*3600), r_b / (header_0['CDELT2']*3600), phi * u.deg)
    mask_total = aperture.to_mask()
    mask_total = np.array(mask_total.to_image(np.shape(moment_0)), dtype = float)
    
    # annular outer mask
    mask_outer = mask_total - mask_inner
    mask_outer[mask_outer < 0] = 0 # to avoid over-subtraction if inner radius is larger than r_b

    f_inner = len(moment_0[(moment_0>0) & (mask_inner>0)])/len(moment_0[(mask_inner>0)])
    f_outer = len(moment_0[(moment_0>0) & (mask_outer>0)])/len(moment_0[(mask_outer>0)])

    return f_inner, f_outer