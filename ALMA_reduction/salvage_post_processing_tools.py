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

def sdss_cutout(objID,band,npix,outdir):

    print('sdss')

    with tempfile.TemporaryDirectory() as wdir:
        # move to working directory defined above
        os.chdir(wdir)
    
        objID = str(objID)
        # first check to see what has already been made in output directory
        gimage_name = outdir+'/'+objID+'_SDSS_'+str(npix)+'x'+str(npix)+'-'+band+'.fits'
        # if this file exists, then the script has already created images for this objID
        if os.access(gimage_name,0):
            return
        
        procflag = -1
        if procflag == -1:
            try:
                # query skyserver for values to avoid table dependencies
                query = ['SELECT p.objID,f.run,f.rerun,f.camcol,f.field,',
                         'f.aa_{b},f.kk_{b},f.airmass_{b},p.colc_{b},p.rowc_{b},'.format(b=band),
                         'f.sky_{b},f.skysig_{b}'.format(b=band),
                         'FROM Field as f JOIN PhotoObjAll as p',
                         'ON (f.run=p.run AND f.rerun=p.rerun AND f.camcol=p.camcol AND f.field=p.field)',
                         'WHERE p.objID = {}'.format(objID)]
                lines = sqlcl.query(str.join(' ',query)).readlines()
                vals = lines[1].decode("utf-8").split('\n')[0].split(',')
                # parse for field and location info
                run = int(vals[1])
                rerun = int(vals[2])
                camcol = int(vals[3])
                field = int(vals[4])
                zeropoint = float(vals[5])
                ext_coeff = float(vals[6])
                airmass = float(vals[7])
                colc = float(vals[8])
                rowc = float(vals[9])
                sky = float(vals[10])*1e9*0.396127**2
                skysig = float(vals[11])*sky*np.log(10)/2.5
                tmp_field = str(field)
                tmp_run = str(run)
                # temporary field and run names for DAS path
                while len(tmp_field) < 4: tmp_field = "0"+tmp_field
                while len(tmp_run) < 6: tmp_run = '0'+tmp_run
            except:
                print('\n Could not locate objID {} on skyserver... continue \n'.format(objID))
                return
    
        
        #############################################
        # must now fetch images from the SDSS server
        # failure --> procflag = 2
        #############################################
        if procflag == -1:
    
            # Path definitions for DR7 DAS
            tmp_run = str(run)
            while len(tmp_run) < 6: tmp_run = '0'+tmp_run
            sdss_path = "imaging/"+str(run)+"/"+str(rerun)+"/objcs/"+str(camcol)+"/"
            sdss_path_c = "imaging/"+str(run)+"/"+str(rerun)+"/corr/"+str(camcol)+"/"
            # SDSS DR7 corrected Image
            corr_name_target = "fpC-"+tmp_run+"-"+band+str(camcol)+"-"+tmp_field+".fit"
            corr_name = "fpC-"+tmp_run+"-"+band+str(camcol)+"-"+tmp_field+"-"+str(objID)+".fit"
            try:
                wgetcmd = 'wget -O '+corr_name+'.gz http://das.sdss.org/'+sdss_path_c+corr_name_target+'.gz'
                os.system(wgetcmd)
                os.system("gunzip "+corr_name+'.gz')
                if not os.access(corr_name,0): procflag = 2
            except:
                raise
                procflag = 2
    
        ############################################
        # build dr7 science image from dr7 corrected image
        # failure --> procflag = 20
        ############################################
        if procflag == -1:
            try:
                sci_xc = int(np.around((npix-1.0)/2.0))
                sci_yc = sci_xc
                sci_nx = npix
                sci_ny = npix
                
                try:
                    corrim = fits.open(corr_name)
                except:
                    raise Exception('Could not open corrected image...')
                
                # image data from corrected image
                corrim_data = corrim[0].data.astype(float)
                # remove softbias
                softbias = float(corrim[0].header['SOFTBIAS'])
                corrim_data -= softbias
                # calibrate to AB nanomaggies
                corrim_data *= 10**(0.4*(zeropoint+ext_coeff*airmass+22.5))/53.907
                # corrected image dimensions
                corrim_nx = corrim[0].header.get('NAXIS1')
                corrim_ny = corrim[0].header.get('NAXIS2')
                # location of galaxy centroid in corrected image
                corrim_xc = int(colc) - 1
                corrim_yc = int(rowc) - 1
                galim = fits.HDUList()
                hdu = fits.PrimaryHDU()
                hdu_data = np.zeros((sci_ny,sci_nx))
                for j in range(sci_ny):
                    for i in range(sci_nx):
                        ii = i - sci_xc + corrim_xc
                        jj = j - sci_yc + corrim_yc
                        if ii >= 0 and ii <= corrim_nx-1 and jj >= 0 and jj <= corrim_ny-1:
                            hdu_data[j,i] = corrim_data[jj,ii]
                        else:
                            hdu_data[j,i] = -9999.99
                
                hdu.data = hdu_data
                # add book-keeping info to output image header
                hdu.header.set('INPUT_IM',corr_name_target,comment='SDSS corrected image')
                hdu.header.set('RUN',run,comment='SDSS Run')
                hdu.header.set('RERUN',rerun,comment='SDSS Rerun')
                hdu.header.set('CAMCOL',camcol,comment='SDSS Camcol')
                hdu.header.set('FIELD',field,comment='SDSS Field')
                hdu.header.set('BAND',band,comment='SDSS Band')
                hdu.header.set('BUNIT','AB nanomaggies',comment='Pixel Units')
                hdu.header.set('SCALE',0.396127,comment='Scale [arcsec/pixel]')
                hdu.header.set('AA',zeropoint,comment='Zeropoint')
                hdu.header.set('KK',ext_coeff,comment='Atm. Extinction Coefficient')
                hdu.header.set('AMASS',airmass,comment='Atm. Airmass')
                hdu.header.set('XCENTER',colc,comment='Object x-position on SDSS corrected image')
                hdu.header.set('YCENTER',rowc,comment='Object y-position on SDSS corrected image')
                hdu.header.set('SKY',sky,comment='Average sky in full SDSS field [nanomaggies]')
                hdu.header.set('SKYSIG',skysig,comment='Average sky uncertainty per pixel [nanomaggies]')
                galim.append(hdu)
                if os.access(gimage_name,0): os.remove(gimage_name)
                galim.writeto(gimage_name)
                corrim.close()
                        
            except:
                raise
                print('\n',procflag,'\n')
                pass
    
        try:
            if os.access(corr_name,0): os.remove(corr_name)
        except:
            os.system('rm fpC*')
            raise Exception('Could not remove corrected image.\n Exiting to avoid problems...\n')


def demo_PHANGS_moments_annuli(imagename, z, r_outer, ID):

    #r_outer = r_outer * 0.396

    try:

        ## load moments
        moment_0 = pyfits.getdata(imagename.replace('MOMENTTYPE', 'strict_mom0'))
        moment_1 = pyfits.getdata(imagename.replace('MOMENTTYPE', 'strict_mom1'))
        moment_2 = pyfits.getdata(imagename.replace('MOMENTTYPE', 'strict_ew'))
        
        header_0 = pyfits.getheader(imagename.replace('MOMENTTYPE', 'strict_mom0'))
        header_1 = pyfits.getheader(imagename.replace('MOMENTTYPE', 'strict_mom1'))
        header_2 = pyfits.getheader(imagename.replace('MOMENTTYPE', 'strict_ew'))
        
    except:
        print(imagename)
        return

    print('\'BMAJ\' from header: ', header_0['BMAJ']*3600)
    
    ## plot moments
    
    Npix = header_0['NAXIS1']
    wpix = 25
    
    min_x, max_x = int(Npix/2 - wpix), int(Npix/2 + wpix)
    min_y, max_y = int(Npix/2 - wpix), int(Npix/2 + wpix)
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 3))

    axes[0].set_title(header_0['BTYPE'], fontsize = 11)
    #contour = axes[0].contour(moment_0[min_x: max_x, min_y: max_y], levels = [KkmsLim], zorder = 10, cmap='Reds')
    #plt.clabel(contour, inline=1, fontsize=10)
    demo_mom0 = moment_0[min_x: max_x, min_y: max_y].copy()
    demo_mom0[demo_mom0<=0] = np.nan
    out = axes[0].imshow(demo_mom0, origin = 'lower', cmap = 'cividis')
    plt.colorbar(out, label=header_0['BUNIT'], shrink = 0.85)

    # plot inner radius
    centre = (wpix,wpix)
    radius = 1.5 / (header_0['CDELT2']*3600)
    circle = Circle(centre, radius, fill = False, color = 'tab:red', lw = 2)
    axes[0].add_patch(circle)

    # plot outer radius
    centre = (wpix,wpix)
    radius_outer = r_outer / (header_0['CDELT2']*3600)
    circle = Circle(centre, radius_outer, fill = False, color = 'tab:pink', lw = 2, ls = '--')
    axes[0].add_patch(circle)

    # plot beam
    centre = (5,5)
    radius_beam = header_0['BMAJ']*3600 / 2 / (header_0['CDELT2']*3600)
    circle = Circle(centre, radius_beam, fill = False, color = 'black', lw = 1.2)
    axes[0].add_patch(circle)
    
    axes[1].set_title(header_1['BTYPE'], fontsize = 11)
    out = axes[1].imshow(moment_1[min_x: max_x, min_y: max_y] - np.nanmedian(moment_1[min_x: max_x, min_y: max_y]), origin = 'lower', cmap = 'coolwarm')
    plt.colorbar(out, label=header_1['BUNIT'], shrink = 0.85)

    axes[2].set_title(header_2['BTYPE'], fontsize = 11)
    out = axes[2].imshow(moment_2[min_x: max_x, min_y: max_y], origin = 'lower', cmap = 'magma')
    plt.colorbar(out, label=header_2['BUNIT'], shrink = 0.85)

    plt.show()

        
    ## calculate MH2
    
    arcsec_per_pix = header_0['CDELT2']*3600 # "/pix
    pc_per_arcsec = (cosmo.arcsec_per_kpc_proper(z=z).value / 1e3) ** -1
    pc_per_pix = pc_per_arcsec * arcsec_per_pix

    L_CO = np.nansum(np.nansum(moment_0 * pc_per_pix**2 ))
    a_CO = 4.35 #(K km s−1 pc2)^−1 # from Lin+20

    M_H2 = L_CO * a_CO
    
    
    ## calculate error on MH2
    
    moment_0_err = pyfits.getdata(imagename.replace('MOMENTTYPE', 'strict_emom0'))
    #L_CO_err = np.nansum(np.sqrt(np.nansum( (moment_0 * pc_per_pix**2)**2 ) ) / np.sqrt(len(moment_0.flatten())))
    L_CO_err = np.nansum(np.sqrt(np.nansum( (moment_0_err * pc_per_pix**2)**2 ) ))
    M_H2_err = L_CO_err * a_CO
    
    #print('log(MH2) = ', np.log10(M_H2), ' +/- ', np.log10(M_H2_err))
    print(f'log(MH2) = {np.log10(M_H2): .5f} +/- {np.log10(M_H2) - np.log10(M_H2 - M_H2_err): .5f}')
    #print('log(MH2) = ', np.log10(M_H2), ' +/- ', np.log10(M_H2 + M_H2_err) - np.log10(M_H2))
    
    
    
    ## calculate MH2 (inner)
    
    center = PixCoord(wpix,wpix)
    radius = 1.5 / (header_0['CDELT2']*3600)
    aperture = CirclePixelRegion(center, radius)
    
    mask = aperture.to_mask(mode='exact')
    moment_0_inner = mask.multiply(moment_0[min_x: max_x, min_y: max_y])
    moment_0_inner_err = mask.multiply(moment_0_err[min_x: max_x, min_y: max_y])

    L_CO_inner = np.nansum(np.nansum(moment_0_inner * pc_per_pix**2 ))
    L_CO_inner_err = np.nansum(np.sqrt(np.nansum( (moment_0_inner_err * pc_per_pix**2)**2 ) ))
    
    a_CO = 4.35 #(K km s−1 pc2)^−1 # from Lin+20

    M_H2_inner = L_CO_inner * a_CO
    M_H2_inner_err = L_CO_inner_err * a_CO
    
    print(f'log(MH2) (inner) = {np.log10(M_H2_inner): .5f} +/- {np.log10(M_H2_inner) - np.log10(M_H2_inner - M_H2_inner_err): .5f}')


    ## calculate MH2 (outer)
    
    center = PixCoord(wpix,wpix)
    radius_inner = 1.5 / (header_0['CDELT2']*3600)
    radius_outer = r_outer / (header_0['CDELT2']*3600)
    #aperture = CircleAnnulusPixelRegion(center, radius_inner, radius_outer)
    aperture = CirclePixelRegion(center, radius_outer)

    mask_outer = aperture.to_mask()
    #boolean defeats the purpose of half values
    #mask_outer = np.array(mask_outer.to_image(np.shape(moment_0[min_x: max_x, min_y: max_y])).data, dtype = 'bool') & (~np.array(mask.to_image(np.shape(moment_0[min_x: max_x, min_y: max_y])).data, dtype = 'bool'))
    mask_outer = np.array(mask_outer.to_image(np.shape(moment_0[min_x: max_x, min_y: max_y])).data, dtype = float) - np.array(mask.to_image(np.shape(moment_0[min_x: max_x, min_y: max_y])).data, dtype = float)
    
    #moment_0_outer = mask_outer.multiply(moment_0[min_x: max_x, min_y: max_y])
    #moment_0_outer_err = mask_outer.multiply(moment_0_err[min_x: max_x, min_y: max_y])
    moment_0_outer = mask_outer * moment_0[min_x: max_x, min_y: max_y]
    moment_0_outer_err = mask_outer * moment_0_err[min_x: max_x, min_y: max_y]

    L_CO_outer = np.nansum(np.nansum(moment_0_outer * pc_per_pix**2 ))
    L_CO_outer_err = np.nansum(np.sqrt(np.nansum( (moment_0_outer_err * pc_per_pix**2)**2 ) ))
    
    a_CO = 4.35 #(K km s−1 pc2)^−1 # from Lin+20

    M_H2_outer = L_CO_outer * a_CO
    M_H2_outer_err = L_CO_outer_err * a_CO
    
    print(f'log(MH2) (outer) = {np.log10(M_H2_outer): .5f} +/- {np.log10(M_H2_outer) - np.log10(M_H2_outer - M_H2_outer_err): .5f}')

    print(f'log(MH2) (inner+outer) = {np.log10(M_H2_inner + M_H2_outer): .5f}')

    plt.figure(figsize = (3,3))
    plt.title('Outer Mask')
    plt.imshow(mask_outer, origin = 'lower')
    plt.show()

    # output to file
    out = open('/arc/projects/salvage/ALMA_reduction/gas_masses_Jun18.txt', 'a')
    out.write(f'{ID} {np.log10(M_H2)} {np.log10(M_H2_err)} {np.log10(M_H2_inner)} {np.log10(M_H2_inner_err)} {np.log10(M_H2_outer)} {np.log10(M_H2_outer_err)}\n')
    out.close()
    
    return

def view_image(i):

    # sdss image

    sdss_fpath = '/arc/projects/salvage/SDSS_images/'
    #npix = int(25*(header_0['CDELT2']*3600/0.396))
    npix = int(50 / 0.396) # 50" x 50" in pix)

    if not os.path.isfile(sdss_fpath + i + f'_SDSS_{npix}x{npix}-r.fits'):
    
        sdss_cutout(i, 'r', npix, sdss_fpath)
        
        with fits.open(sdss_fpath + i + f'_SDSS_{npix}x{npix}-r.fits', mode='update') as hdul:
            hdul[0].data -= hdul[0].header['SKY']
            hdul[0].data[hdul[0].data<-500] = 0.0
            hdul.flush()
            
    sdss = fits.getdata(sdss_fpath + i + f'_SDSS_{npix}x{npix}-r.fits')
    wcs_sdss = WCS(fits.getheader(sdss_fpath + i + f'_SDSS_{npix}x{npix}-r.fits'))

    sdss = rotate(sdss, -90)
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    ax.set_title(f'SDSS Image ({i})', fontsize = 11)
    NORM = LogNorm(vmin=np.percentile(sdss[sdss>0], 1),vmax=np.percentile(sdss[sdss>0], 99.99))
    out = ax.imshow(sdss, origin = 'lower', cmap = 'gray_r', norm = NORM)

    plt.show()
    
    return

def demo_moments_on_image(imagename, z, i, r_outer):

    try:
        ## load moment 0 map
        moment_0 = pyfits.getdata(imagename.replace('MOMENTTYPE', 'strict_mom0'))
        header_0 = pyfits.getheader(imagename.replace('MOMENTTYPE', 'strict_mom0'))

    except:
        print('Could not open moment 0 map for the following:', imagename)
        return
    
    wcs = WCS(header_0)


    # sdss image

    sdss_fpath = '/arc/projects/salvage/SDSS_images/'
    #npix = int(25*(header_0['CDELT2']*3600/0.396))
    npix = int(50 / 0.396) # 50" x 50" in pix)

    if not os.path.isfile(sdss_fpath + i + f'_SDSS_{npix}x{npix}-r.fits'):
    
        sdss_cutout(i, 'r', npix, sdss_fpath)
        
        with fits.open(sdss_fpath + i + f'_SDSS_{npix}x{npix}-r.fits', mode='update') as hdul:
            hdul[0].data -= hdul[0].header['SKY']
            hdul[0].data[hdul[0].data<-500] = 0.0
            hdul.flush()
            
    sdss = fits.getdata(sdss_fpath + i + f'_SDSS_{npix}x{npix}-r.fits')
    wcs_sdss = WCS(fits.getheader(sdss_fpath + i + f'_SDSS_{npix}x{npix}-r.fits'))
    wcs_alma = wcs

    #sdss = rotate(sdss, -90)
    
    
    ## plot moments
    
    Npix = header_0['NAXIS1']
    wpix = int(25 / (header_0['CDELT2']*3600)) # 30" x 30" in pix)
    
    min_x, max_x = int(Npix/2 - wpix), int(Npix/2 + wpix)
    min_y, max_y = int(Npix/2 - wpix), int(Npix/2 + wpix)
    '''
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.set_title('CO(1-0) Flux', fontsize = 11)
    #contour = axes[0].contour(moment_0[min_x: max_x, min_y: max_y], levels = [KkmsLim], zorder = 10, cmap='Reds')
    #plt.clabel(contour, inline=1, fontsize=10)
    out = ax.imshow(moment_0[min_x: max_x, min_y: max_y], origin = 'lower', cmap = 'gray_r')
    plt.colorbar(out, label=header_0['BUNIT'], shrink = 0.85)
    
    centre = (wpix,wpix)
    radius_outer = r_outer / (header_0['CDELT2']*3600)
    circle = Circle(centre, radius_outer, fill = False, color = 'tab:green')
    ax.add_patch(circle)
    
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec.')

    #plt.savefig(f'fancy_image_{z}.png', bbox_inches = 'tight', dpi = 200)
    plt.show()

    print(wpix)
    '''
    #fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    ax = plt.subplot(projection = wcs_alma)

    #print(ax.get_transform("world"))
    #print(ax.get_transform(wcs_sdss))
    #print(type(WCS(header_0)), type(wcs_sdss))
    #print(ax.get_transform(WCS(header_0)))

    ax.set_title('SDSS Image', fontsize = 11)
    #contour = ax.contour(moment_0[min_x: max_x, min_y: max_y], levels = 15, zorder = 10, cmap='Reds', alpha = 0.5)
    contour = ax.contour(moment_0[min_x: max_x, min_y: max_y], levels = 5, zorder = 10, cmap='YlOrRd', alpha = 0.5, linewidths = 2, transform = ax.get_transform(WCS(header_0)))
    #plt.clabel(contour, inline=1, fontsize=10)
    NORM = LogNorm(vmin=np.percentile(sdss[sdss>0], 1),vmax=np.percentile(sdss[sdss>0], 99.99))
    out = ax.imshow(sdss, origin = 'lower', cmap = 'gray_r', extent = (0, 2*wpix, 0, 2*wpix), norm = NORM)
    #plt.colorbar(out, shrink = 0.85)

    centre = (wpix,wpix)
    radius_outer = r_outer / (header_0['CDELT2']*3600)
    circle = Circle(centre, radius_outer, fill = False, color = 'tab:pink', lw = 2, ls = '--')
    ax.add_patch(circle)

    #plt.savefig(f'fancy_image_{z}.png', bbox_inches = 'tight', dpi = 200)
    plt.show()
    
    return

def demo_image_and_moment_separate(imagename, z, ra, dec, r_outer, ID):

    try:

        ## load moment0
        moment_0 = pyfits.getdata(imagename.replace('MOMENTTYPE', 'strict_mom0'))
        header_0 = pyfits.getheader(imagename.replace('MOMENTTYPE', 'strict_mom0'))
        
    except:
        print(imagename)
        return

    print('\'BMAJ\' from header: ', header_0['BMAJ']*3600)

    wcs = WCS(header_0)

    fig = plt.figure(figsize=(14, 10))
    axes = [plt.subplot(121), plt.subplot(122, projection = wcs)]

    # PLOT IMAGE FIRST
    
    i = ID.copy()

    sdss_fpath = '/arc/projects/salvage/SDSS_images/'
    #npix = int(25*(header_0['CDELT2']*3600/0.396))
    npix = int(50 / 0.396) # 50" x 50" in pix)

    if not os.path.isfile(sdss_fpath + i + f'_SDSS_{npix}x{npix}-r.fits'):
    
        sdss_cutout(i, 'r', npix, sdss_fpath)
        
        with fits.open(sdss_fpath + i + f'_SDSS_{npix}x{npix}-r.fits', mode='update') as hdul:
            hdul[0].data -= hdul[0].header['SKY']
            hdul[0].data[hdul[0].data<-500] = 0.0
            hdul.flush()
            
    sdss = fits.getdata(sdss_fpath + i + f'_SDSS_{npix}x{npix}-r.fits')
    sdss = rotate(sdss, -90)
    
    axes[0].set_title(f'SDSS Image ({i})', fontsize = 11)
    NORM = LogNorm(vmin=np.percentile(sdss[sdss>0], 1),vmax=np.percentile(sdss[sdss>0], 99.99))
    out = axes[0].imshow(sdss, origin = 'lower', cmap = 'gray_r', norm = NORM)


    # plot inner radius
    wpix = np.shape(sdss)[0]/2
    centre = (wpix,wpix)
    radius = 1.5 / 0.396
    circle = Circle(centre, radius, fill = False, color = 'tab:red', lw = 2.6)
    axes[0].add_patch(circle)

    # plot outer radius
    radius_outer = r_outer / 0.396
    circle = Circle(centre, radius_outer, fill = False, color = 'tab:pink', lw = 2.6, ls = '--')
    axes[0].add_patch(circle)

    axes[0].tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    

    # PLOT MOMENT0 NEXT
    
    
    
    Npix = header_0['NAXIS1']
    wpix = 25
    
    min_x, max_x = int(Npix/2 - wpix), int(Npix/2 + wpix)
    min_y, max_y = int(Npix/2 - wpix), int(Npix/2 + wpix)
    
    

    axes[1].set_title('ALMA CO(1-0) Moment Map', fontsize = 11)
    demo_mom0 = moment_0[min_x: max_x, min_y: max_y].copy()
    demo_mom0[demo_mom0<=0] = np.nan
    out = axes[1].imshow(demo_mom0, origin = 'lower', cmap = 'cividis')
    plt.colorbar(out, label=header_0['BUNIT'], shrink = 0.55)

    # plot inner radius
    centre = (wpix,wpix)
    radius = 1.5 / (header_0['CDELT2']*3600)
    circle = Circle(centre, radius, fill = False, color = 'tab:red', lw = 2)
    axes[1].add_patch(circle)

    # plot outer radius
    centre = (wpix,wpix)
    radius_outer = r_outer / (header_0['CDELT2']*3600)
    circle = Circle(centre, radius_outer, fill = False, color = 'tab:pink', lw = 2, ls = '--')
    axes[1].add_patch(circle)

    add_beam(axes[1], header_0, fill = False, color = 'k')
    add_scalebar(axes[1], 10/3600, label = '10"')

    for ax in axes[1:]:
        dec = ax.coords[0]
        ra  = ax.coords[1]

        dec.set_axislabel('R.A.')
        ra.set_axislabel('Dec.')

        dec.set_major_formatter('d.ddd')
        ra.set_major_formatter('d.ddd')

        #dec.set_ticks(number=4)
        #ra.set_ticks(number=4)

        dec.set_ticklabel(exclude_overlapping=True)
        ra.set_ticklabel(exclude_overlapping=True)

        dec.display_minor_ticks(True)
        ra.display_minor_ticks(True)
        
        dec.set_minor_frequency(10)
        ra.set_minor_frequency(10)


        ax.tick_params(axis = 'both', which = 'both', direction = 'in')

    

    #plt.savefig(f'./figures/image_and_moment_{ID}.png', bbox_inches = 'tight', dpi = 200)

    plt.show()
    
    return

def demo_PHANGS_moments_annuli_wcs2(imagename, z, ra, dec, r_outer, ID):

    try:

        ## load moments
        moment_0 = pyfits.getdata(imagename.replace('MOMENTTYPE', 'strict_mom0'))
        moment_1 = pyfits.getdata(imagename.replace('MOMENTTYPE', 'strict_mom1'))
        moment_2 = pyfits.getdata(imagename.replace('MOMENTTYPE', 'strict_ew'))
        
        header_0 = pyfits.getheader(imagename.replace('MOMENTTYPE', 'strict_mom0'))
        header_1 = pyfits.getheader(imagename.replace('MOMENTTYPE', 'strict_mom1'))
        header_2 = pyfits.getheader(imagename.replace('MOMENTTYPE', 'strict_ew'))
        
    except:
        print(imagename)
        return

    print('\'BMAJ\' from header: ', header_0['BMAJ']*3600)
    
    ## plot moments

    wcs = WCS(header_0)

    center_coord = SkyCoord(ra, dec, unit="deg") 
    center_x, center_y = wcs.world_to_pixel(center_coord)
    centre = (center_x, center_y)
    
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 3), subplot_kw={'projection': wcs})

    axes[0].set_title(header_0['BTYPE'], fontsize = 11)
    demo_mom0 = moment_0.copy()
    demo_mom0[demo_mom0<=0] = np.nan
    out = axes[0].imshow(demo_mom0, origin = 'lower', cmap = 'cividis')
    plt.colorbar(out, label=header_0['BUNIT'], shrink = 0.85)

    # plot inner radius
    radius = 1.5 / (header_0['CDELT2']*3600)
    circle = Circle(centre, radius, fill = False, color = 'tab:red', lw = 2)
    axes[0].add_patch(circle)

    # plot outer radius
    radius_outer = r_outer / (header_0['CDELT2']*3600)
    circle = Circle(centre, radius_outer, fill = False, color = 'tab:pink', lw = 2, ls = '--')
    axes[0].add_patch(circle)

    add_beam(axes[0], header_0, fill = False, color = 'k')
    add_scalebar(axes[0], 5/3600, label = '5"')
    
    axes[1].set_title(header_1['BTYPE'], fontsize = 11)
    out = axes[1].imshow(moment_1 - np.nanmedian(moment_1), origin = 'lower', cmap = 'coolwarm')
    plt.colorbar(out, label=header_1['BUNIT'], shrink = 0.85)

    axes[2].set_title(header_2['BTYPE'], fontsize = 11)
    out = axes[2].imshow(moment_2, origin = 'lower', cmap = 'magma')
    plt.colorbar(out, label=header_2['BUNIT'], shrink = 0.85)

    ## plot real centre test
    axes[0].scatter(center_x, center_y, marker = 'x', color = 'k', alpha = 0.5)
    ##


    for ax in axes:
        dec = ax.coords[0]
        ra  = ax.coords[1]

        dec.set_axislabel('R.A.')
        ra.set_axislabel('Dec.')

        dec.set_major_formatter('d.ddd')
        ra.set_major_formatter('d.ddd')

        #dec.set_ticks(number=4)
        #ra.set_ticks(number=4)

        dec.set_ticklabel(exclude_overlapping=True)
        ra.set_ticklabel(exclude_overlapping=True)

        dec.display_minor_ticks(True)
        ra.display_minor_ticks(True)
        
        dec.set_minor_frequency(10)
        ra.set_minor_frequency(10)

        ax.tick_params(axis = 'both', which = 'both', direction = 'in')

    plt.show()

    
    return

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
    - rms:         Root mean square error of spectrum, calculated from extant 200 km/s of spectrum. 
    '''

    # generate a window over which the flux will be measured
    rms_mask = (vel < (np.nanmin(vel) + (200 * u.km / u.s))) | (vel > (np.nanmax(vel) - (200 * u.km / u.s)))                                              
    rms = np.nanstd(spec[rms_mask])

    mask = ~rms_mask & (spec>0)

    # compute SNR across the spectrum
    snr = spec / rms

    # initialize core mask
    mask_core = snr > snr_hi
    
    for iiter in range(nchan_hi-1):
        mask_core &= np.roll(mask_core, 1, 0)
    
    for iiter in range(nchan_hi-1):
        mask_core |= np.roll(mask_core, -1, 0)

    mask_core &= mask
    
    # initialize wing mask
    mask_wing = snr > snr_lo
    
    for iiter in range(nchan_lo-1):
        mask_wing &= np.roll(mask_wing, 1, 0)
    
    for iiter in range(nchan_lo-1):
        mask_wing |= np.roll(mask_wing, -1, 0)

    mask_wing &= mask

    # dilate core mask inside wing mask
    mask_signal = binary_dilation(
        mask_core, iterations=0, mask=mask_wing)

    if expand_by_nchan > 0:

        for iiter in range(expand_by_nchan):
            tempmask = np.roll(mask_signal, 1, axis=0)
            mask_signal |= tempmask
            tempmask = np.roll(mask_signal, -1, axis=0)
            mask_signal |= tempmask

    mask_signal &= mask

    return mask_signal, rms


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

def MH2_from_cube_phys_old(ID, z, ra, dec, r_inner, r_outer, mask_type = 'total', a_CO = 4.35):

    file         = f'/arc/projects/salvage/ALMA_reduction/phangs_pipeline/imaging/{ID}/{ID}_12m_co10.fits'
    header_image = pyfits.getheader(file)
    
    # import cube, convert to K
    cube = SpectralCube.read(file, format='fits').to(u.K).with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=(115.27120180 * u.GHz)/(1+z))
    
    # compute pc_per_pix from redshift and pixel scale
    arcsec_per_pix = header_image['CDELT2']*3600 # "/pix
    pc_per_arcsec = (cosmo.arcsec_per_kpc_proper(z=z).value / 1e3) ** -1
    pc_per_pix = pc_per_arcsec * arcsec_per_pix
    
    if mask_type == 'inner':
    
        # find central pixel from SDSS ra and dec
        wcs = WCS(header_image)[0]
        center_coord = SkyCoord(ra, dec, unit="deg") 
        center_x, center_y = wcs.world_to_pixel(center_coord)
    
        # generate a circular aperture
        center = PixCoord(center_x, center_y)
        radius = r_inner / pc_per_pix
        aperture = CirclePixelRegion(center, radius)
    
        # convert aperture to a mask and apply to the moment map
        # mode = 'exact' uses partial contribution from edge pixels to simulate a perfect circle
        mask = aperture.to_mask(mode='exact')
        
        # make mask an array with same dimensions as a cube slice
        mask = np.array(mask.to_image(np.shape(cube[0,:,:])).data, dtype = float)
    
        # apply mask to cube
        masked_cube = cube * mask
    
    if mask_type == 'outer':
    
        # both circular apertures will use the same center
        wcs = WCS(header_image)[0]
        center_coord = SkyCoord(ra, dec, unit="deg") 
        center_x, center_y = wcs.world_to_pixel(center_coord)
        center = PixCoord(center_x, center_y)
    
        # generate an inner circular aperture with mode = 'exact'
        radius_inner = r_inner / pc_per_pix
        aperture = CirclePixelRegion(center, radius_inner)
        mask_inner = aperture.to_mask(mode='exact')
        mask_inner = np.array(mask_inner.to_image(np.shape(cube[0,:,:])).data, dtype = float)
    
        # generate an outer circular aperture
        radius_outer = r_outer / pc_per_pix
        aperture = CirclePixelRegion(center, radius_outer)
        mask_outer = aperture.to_mask()
        mask_outer = np.array(mask_outer.to_image(np.shape(cube[0,:,:])).data, dtype = float)
    
        mask_outer = mask_outer - mask_inner
    
        # apply mask to cube
        masked_cube = cube * mask_outer

    if mask_type == 'total':

        masked_cube = cube
    
    plt.figure(figsize = (8,5))
    
    masked_cube[:,int(header_image['CRPIX1']),int(header_image['CRPIX1'])].quicklook(label = 'Central Spectrum')
    
    # replace nans with 0's, don't want them adding to the stack...
    #masked_cube = masked_cube.with_fill_value(0.)  still nans
    masked_cube = masked_cube.apply_numpy_function(np.nan_to_num, fill=0)
    
    # stack the masked cube spectra
    stack = np.nansum(masked_cube, axis = (1,2))
    
    
    plt.plot(cube.spectral_axis, stack, ds = 'steps-mid', label = 'Integrated Spectrum')
    
    plt.legend(fancybox = True)
    plt.title(f'Integrated Spectrum ({mask_type})')
    
    plt.show()
    
    
    # total summed masked cube should give mass
    L_CO = np.nansum(stack[(cube.spectral_axis < (np.nanmax(cube.spectral_axis) - (100 * u.km / u.s))) & (cube.spectral_axis > (np.nanmin(cube.spectral_axis) + (100 * u.km / u.s)))]) * pc_per_pix**2 * header_image['CDELT3']/-1e3
    M_H2 = L_CO * a_CO

    if M_H2 < 0:

        M_H2 = 0
    
    L_CO_err = np.nanstd(stack[(cube.spectral_axis > (np.nanmax(cube.spectral_axis) - (100 * u.km / u.s))) | (cube.spectral_axis < (np.nanmin(cube.spectral_axis) + (100 * u.km / u.s)))]) * pc_per_pix**2 * header_image['CDELT3']/-1e3
    M_H2_err = L_CO_err * a_CO


    return M_H2, M_H2_err


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