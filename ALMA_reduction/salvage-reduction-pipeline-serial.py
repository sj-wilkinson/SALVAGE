import pymysql, os, pandas, alminer, glob, time, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from skaha.session import Session
import nest_asyncio
nest_asyncio.apply()

def get_casa_version(PATH, UID):

    version = 'unknown'
    calib = 'manual'
    method = None

    # walk through the directory structure for this SDSS object observed for this ALMA Project 
    for root, dirs, files in os.walk(PATH):

        # if xml files are found, consider this object pipeline calibrated
        if len(glob.glob(f'{root}/*xml')) > 0:
            calib='pipeline'

    ### if a pipeline_manifest.xml file exists, we can pull the appropriate casa version from it ###
    if len(glob.glob(f'{PATH}/script/member.uid___{UID}.*image*.pipeline_manifest.xml'))>0:

        # if at least one pipeline_manifest.xml exists, it shouldn't matter which one you use, so just take the first
        file_to_check = glob.glob(f'{PATH}/script/member.uid___{UID}.*image*.pipeline_manifest.xml')[0]

        # read in the lines of the xml file ...
        with open(file_to_check, 'r') as file:
            content = file.readlines()

        # search through the lines for the casa version name
        for line in content:

            # check if current line has < > formatting ...
            if len(line.split('<')) < 2:
                continue

            split_line = line.split('<')[1].split('=')

            # if there was a '=', it may be indicating the casa version
            if len(split_line) > 1:

                # if before the '=' was 'casaversion name'
                if split_line[0] == 'casaversion name':

                    # then save what comes after as the casa version
                    version = line.split('<')[1].split('=')[1].split('\"')[1]
                    method = '.xml files'


    ### if there are no .xml files, try using the casa logs ###
    elif len(glob.glob(f'{PATH}/log/casa*.log'))>0:

        file_to_check = glob.glob(f'{PATH}/log/casa*.log')[0]

        # read in the lines of the log file ...
        with open(file_to_check, 'r') as file:
            content = file.readlines()

        # search through the lines for the casa version name
        for line in content:

            split_line = line.split('CASA Version')

            # if the current line was split by the above text, it must be present
            if len(split_line)>1:

                # extract casa version from line
                version = split_line[1].split(' ')[1].split('-')[0]
                version = convert_casa_version_name(version)
                method = 'CASA logs'

    ### if there are no .xml files or casa log files, try searching in scriptForImaging.py for their casa version check ###
    elif len(glob.glob(f'{PATH}/script/*scriptForImaging.py'))>0:

        file_to_check = glob.glob(f'{PATH}/script/*scriptForImaging.py')[0]

        # read in the lines of the log file ...
        with open(file_to_check, 'r') as file:
            content = file.readlines()

        # search through the lines for the casa version name
        for line in content:

            split_line = line.split('ERROR: PLEASE USE THE SAME VERSION OF CASA THAT YOU USED FOR GENERATING THE SCRIPT: ')

            # if the current line was split by the above text, it must be present
            if len(split_line)>1:

                    # extract casa version from line
                    version = split_line[1].split("\'")[0]
                    method = 'scriptForImaging.py'
        
    else:

        print('No /script/*.pipeline_manifest.xml files, /log/casapy*.log files, or /script/*scriptForImaging.py.')
        method = 'failed'
        
    return version, calib, method


def convert_casa_version_name(version_in):

    # translation here: https://almascience.nrao.edu/processing/science-pipeline#version
    
    if version_in == '5.6.42831':
        version_out='5.6.1'
    elif version_in == '5.4.42866':
        version_out='5.4.0'
    elif version_in == '5.1.40896':
        version_out='5.1.1'
    elif version_in == '4.7.39732':
        version_out='4.7.2'
    elif version_in == '4.7.38335':
        version_out='4.7.0'
    elif version_in == '4.5.36091':
        version_out='4.5.2'
    elif version_in == '4.5.35996':
        version_out='4.5.1'
    elif version_in == '4.3.32491':
        version_out='4.3.1'
    elif version_in == '4.2.30986':
        version_out='4.2.2'
    else:
        print('Conversion not supported, returning the same version.')
        version_out=version_in

    return version_out

def casa_version_to_canfar_image(version):

    # translation here: https://almascience.nrao.edu/processing/science-pipeline#version

    if version == '6.4.1.12':
        image='images.canfar.net/casa-6/casa:6.4.1-12-pipeline'

    elif version == '6.1.1-15':
        image='images.canfar.net/casa-6/casa:6.1.1-15-pipeline'
    
    elif version == '6.1.1':
        image='images.canfar.net/casa-6/casa:6.1.1-15-pipeline'

    elif version == '5.6.1-8':
        image='images.canfar.net/casa-5/casa:5.6.1-8-pipeline'

    elif version == '5.4.0-70':
        image='images.canfar.net/casa-5/casa:5.4.0-70'
        
    elif version == '5.4.0-68':
        image='images.canfar.net/casa-5/casa:5.4.0-70'
        
    elif version == '5.1.1-5':
        image='images.canfar.net/casa-5/casa:5.1.1-5'

    elif version == '5.1.1':
        image='images.canfar.net/casa-5/casa:5.1.1-5'
        
    elif version == '4.7.2':
        image='images.canfar.net/casa-4/casa:4.7.2'
        
    elif version == '4.7.0':
        image='images.canfar.net/casa-4/casa:4.7.0'

    elif version == '4.6.0':
        print('WARNING: CASA v4.6.0 does not have a pipeline version, using CASA v4.7.0 instead.')
        image='images.canfar.net/casa-4/casa:4.7.0'

    elif version == '4.5.3':
        image='images.canfar.net/casa-4/casa:4.5.3'
        
    elif version == '4.5.2':
        image='images.canfar.net/casa-4/casa:4.5.2'
        
    elif version == '4.5.1':
        image='images.canfar.net/casa-4/casa:4.5.1'
        
    elif version == '4.3.1':
        image='images.canfar.net/casa-4/casa:4.3.1-pipe'
        
    elif version == '4.2.2':
        image='images.canfar.net/casa-4/casa:4.2.2-pipe'

    elif version == '4.2':
        print('WARNING: CASA v4.2 does not have a pipeline version, using CASA v4.2.2 instead.')
        image='images.canfar.net/casa-4/casa:4.2.2-pipe'
        
    else:
        print(f'Your CASA version ({version}) is either not supported or more recent than 6.1.1-15, defaulting to most recent: 6.5.4-9.')
        image = 'images.canfar.net/casa-6/casa:6.5.4-9-pipeline'

        # I keep receiving an error with 6.5.4-9: warnings.warn('PyFITS is deprecated, please use astropy.io.fits', let's try an earlier version?
        #print(f'Your CASA version ({version}) is either not supported or more recent than 6.1.1-15, defaulting to: 6.2.1-7.')
        #image = 'images.canfar.net/casa-6/casa:6.2.1-7-pipeline'   

    return image

###########################################

#### STAGE 0: READ IN SAMPLE TO REDUCE ####

###########################################

fpath = '/arc/projects/salvage/ALMA_reduction/samples/'
#file  =  'salvage_Feb12_sample.txt'
file  =  'salvage_Feb12_sample_mrs_gt_2rp_AGN.txt'

objID_sample, year_sample, name_sample, muid_sample, guid_sample, auid_sample, proj_sample = np.loadtxt(fpath+file, unpack = True, dtype = str, usecols = [0,11,12,13,14,15,16])
z_sample, mass_sample, rpetro_sample, ra_sample, dec_sample, res_sample, mrs_sample, AL_sample, AC_sample, TP_sample = np.loadtxt(fpath+file, unpack = True, dtype = float, usecols = [1,2,3,4,5,6,7,8,9,10])

# galaxies to reduce in this run
argv = sys.argv
min_index = int(argv[-2])
max_index = int(argv[-1])

# downloads that take up ~300-500GB of disk space                                                                                                                              # 587736803470540958 took up 7T in intermediate imaging...
massive_downloads = ['588017703996096564', '588848900431216934', '587727177931817054' , '588848900966514994', '588848899357737008', '587727229448421420', '587741489300766802', '587736803470540958']

rerun_targets = ['587730772799914185'] # picked a random one that should work but is raising error

do_stage1 = False
do_stage2 = False
do_stage3 = True
do_stage4 = True

rerun_only = False
skip_massive_downloads = True
skip_completed = True

# loop over galaxies and launch jobs
for i in np.arange(min_index,max_index):

    ID = objID_sample[i]
    NAME = name_sample[i]
    MUID = muid_sample[i]
    PROJ = proj_sample[i]
    Z    = z_sample[i]
    DIST = cosmo.angular_diameter_distance(Z).value
    RA   = ra_sample[i]
    DEC  = dec_sample[i]

    # choose to only run on problem galaxies from a previous run
    if (ID not in rerun_targets) & rerun_only:
        continue

    # skip if file is known to be prohibitively large
    if (ID in massive_downloads) & skip_massive_downloads:
        print('##################################################################')
        print(f'Skipping {ID} because it is known to require a lot of disk space.')
        print('##################################################################')
        continue

    # skip if file is known to be prohibitively large
    if os.path.exists(f'/arc/projects/salvage/ALMA_reduction/phangs_pipeline/derived/{ID}/{ID}_12m_co10_strict_mom0.fits') & skip_completed:
        print('##############################################################################')
        print(f'Skipping {ID} because it already has a moment 0 map from the PHANGS pipeline.')
        print('##############################################################################')
        continue

    # remove completion flag files for this galaxy
    os.system(f'rm -rf /arc/projects/salvage/ALMA_reduction/salvage_completion_files/{ID}_*_complete.txt')

    #######################################
    
    ##### STAGE 1: DOWNLOAD AND UNZIP #####

    #######################################

    # select appropriate resources
    image = "images.canfar.net/skaha/astroml:24.03"
    cmd = '/arc/projects/salvage/ALMA_reduction/bash_scripts/download_and_unzip_revert.sh'
    ram=4
    cores=2

    if do_stage1:
    
        # launch headless session on the science platform
        session = Session()
        session_id = session.create(
            name  = ID,
            image = image,
            cores = cores,
            ram   = ram,
            kind  = "headless",
            cmd   = cmd,
            args  = f'{ID} {NAME} {PROJ}'
        )
        print("Sesion ID: {}".format(session_id[0]))
    
        # do not continue until bash script has completed
        t = 0
        while not os.path.exists(f'/arc/projects/salvage/ALMA_reduction/salvage_completion_files/{ID}_download_complete.txt'):
            # check every minute
            time.sleep(60)
            t+=1
            print(f'Download has not yet completed. Elapsed time: {t} min.')
    
        print('Download completed. Moving on to calibration.\n')

    else: 
        print('Skipping STAGE 1: DOWNLOAD AND UNZIP\n')

    ########################################

    ##### STAGE 2: RESTORE CALIBRATION #####

    ########################################

    # select appropriate resources
    cmd = '/arc/projects/salvage/ALMA_reduction/bash_scripts/restore_calibration.sh'
    ram=8
    cores=2

    # in some cases the MUID is a number I think?
    try:
        UID =  MUID.split('/')[2] + '_' + MUID.split('/')[3] + '_' + MUID.split('/')[4]
    except:
        print('MUID is in the wrong format.')
        print(UID, type(MUID))
        continue

    # search all directories in the ALMA project folder for the relevant data
    PATH = None
    for root, dirs, files in os.walk(f"/arc/projects/salvage/ALMA_data/{ID}/{PROJ}/"):
        if "member.uid___"+UID in dirs:
            PATH = os.path.join(root, "member.uid___"+UID)
    if PATH == None:
        print()
        print(f'Path to data not found. Skipping this galaxy ({ID}).\n')
        print()
        continue

    # identify and select the appropriate CASA version
    version, calib, method = get_casa_version(PATH, UID)
    image = casa_version_to_canfar_image(version)
    print(ID, version, calib, method, image)

    if do_stage2:
            
        # launch headless session on the science platform
        session = Session()
        session_id = session.create(
            name  = ID,
            image = image,
            cores = cores,
            ram   = ram,
            kind  = "headless",
            cmd   = cmd,
            args  = f'{ID} {PROJ} {PATH}'
        )
    
        print("Sesion ID: {}".format(session_id[0]))
    
        # do not continue until bash script has completed
        t = 0
        while not os.path.exists(f'/arc/projects/salvage/ALMA_reduction/salvage_completion_files/{ID}_calibration_complete.txt'):
            # check every minute
            time.sleep(60)
            t+=1
            print(f'Calibration has not yet completed. Elapsed time: {t} min.')
    
        print('Calibration restored. Moving on to imaging.\n')

    else:

        print('Skipping STAGE 2: RESTORE CALIBRATION\n')

    ############################################

    #### STAGE 3a: PREP FOR PHANGS PIPELINE ####

    ############################################

    key_dir = '/arc/projects/salvage/phangs_imaging_scripts/phangs-alma_keys/'

    ## write function to make measurement set key for single target
    
    ms_key_out = 'ms_file_key_salvage.txt'
    ms_key_tmp = 'ms_file_key_template.txt'
    
    # get template table (formatting with no file paths)
    out = open(key_dir + ms_key_tmp, 'r')
    out_data = out.read()
    out.close()
    
    # wipe old content from output file
    out = open(key_dir + ms_key_out, 'w')
    out.close()

    # initialize string
    out_str = ''

    # prep file path for PHANGS-ALMA pipeline
    ms_root = '/arc/projects/salvage/ALMA_data/'
    ms_filepath = PATH.replace(ms_root, '') # wipe ms_root directory so it can be added separately
    ms_filepath += '/calibrated/' # add calibrated on the end so that it points to the calibrated data

    # until I discover more nuance, take all measurement set files from this directory and let PHANGS deal with them
    ms_files = glob.glob(ms_root + ms_filepath + '*.ms*')

    print(ms_root + ms_filepath, os.path.isdir(ms_root + ms_filepath), len(ms_files))

    if len(ms_files)<1:
        #checking if files are missing because of bug in previous stages or this one...
        print('No MS files found in ', ms_filepath)

        print('CANCEL IMAGING DUE TO MISSING FILES.\n')
        continue

    # check if there are visibilities for this target from another project and add on top if there is...
    #init = len(np.array(objID_in_file)[np.array(objID_in_file)==ID])
    init = 0

    # add each measurement set to ms key
    for j in range(init, init + len(ms_files)):

        ms_path = ms_files[j-init].replace(ms_root, '') # wipe ms_root directory so it can be added separately
        obs_num = j+1 # record different measurement sets as different observations
    
        out_str += f"{ID} {ID}_{obs_num} all 12m {obs_num} {ms_path}\n" 
        # need to accomodate other observations of the same objID?

    # replace placeholder "__DATA__" in template with formatted data
    out_data = out_data.replace('__DATA__', out_str)
        
    # write to new key file
    out = open(key_dir + ms_key_out, 'w')
    out.write(out_data)
    out.close()

    ## write function to make distance key for single target

    dist_key_out = 'distance_key_salvage.txt'
    dist_key_tmp = 'distance_key_template.txt'
    
    # get template table (formatting with no coordinates)
    out = open(key_dir + dist_key_tmp, 'r')
    out_data = out.read()
    out.close()
    
    # initialize string
    out_str = ''
    out_str += f"{ID},{DIST}\n" 
    # need to accomodate other observations of the same objID?
    
    # replace __DATA__ in template with formatted data
    out_data = out_data.replace('__DATA__', out_str)
    
    # write to new key file
    out = open(key_dir + dist_key_out, 'w')
    out.write(out_data)
    out.close()

    ## write function to make target defs key for single target

    targ_key_out = 'target_definitions_salvage.txt'
    targ_key_tmp = 'target_definitions_template.txt'

    # prep for sys vel calculation
    restfreq = 115.27120 * u.GHz  # rest frequency of 12 CO 1-0 in GHz
    freq_to_vel = u.doppler_radio(restfreq)

    # get template table (formatting with no coordinates)
    out = open(key_dir + targ_key_tmp, 'r')
    out_data = out.read()
    out.close()

    # initialize string
    out_str = ''
    
    # format the ra and dec's for html table
    c = SkyCoord(ra= RA*u.degree, dec= DEC*u.degree, frame='icrs')
    coord_str = c.to_string('hmsdms')

    sys_vel = (restfreq/(1+Z)).to(u.km / u.s, equivalencies=freq_to_vel).value
    velwidth = 1000

    out_str += f"{ID} {coord_str} {sys_vel} {velwidth}\n" 
    # need to accomodate other observations of the same objID?
    
    # replace __DATA__ in template with formatted data
    out_data = out_data.replace('__DATA__', out_str)
    
    # write to new key file
    out = open(key_dir + targ_key_out, 'w')
    out.write(out_data)
    out.close()

    #########################################
    
    ##### STAGE 3b: RUN PHANGS PIPELINE #####

    #########################################

    if do_stage3:

        # select appropriate resources
        #image = "images.canfar.net/casa-6/casa:6.5.6-22"
        image = "images.canfar.net/casa-6/casa:6.5.4-9-pipeline"
        cmd = '/arc/projects/salvage/ALMA_reduction/bash_scripts/run_PHANGS_pipeline.sh'
        ram=16
        cores=2
        
        # launch headless session on the science platform
        session = Session()
        session_id = session.create(
            name  = ID,
            image = image,
            cores = cores,
            ram   = ram,
            kind  = "headless",
            cmd   = cmd,
            args  = f'{NAME} {ID}'
        )
    
        print("Sesion ID: {}".format(session_id[0]))
    
        # do not continue until bash script has completed
        t = 0
        while not os.path.exists(f'/arc/projects/salvage/ALMA_reduction/salvage_completion_files/{ID}_imaging_complete.txt'):
            # check every minute
            time.sleep(60)
            t+=1
            print(f'Imaging has not yet completed. Elapsed time: {t} min.')
    
        print('Imaging completed. Moving on to derived data.\n')

    else:

        print('Skipping STAGE 3: RUN PHANGS PIPELINE\n')

    ################################################

    ##### STAGE 4: RUN PHANGS DERIVED PIPELINE #####

    ################################################

    if do_stage4:

        # select appropriate resources
        image = "images.canfar.net/skaha/astroml:latest"
        cmd = '/arc/projects/salvage/ALMA_reduction/bash_scripts/run_PHANGS_moments.sh'
        ram=4
        cores=2
    
        # launch headless session on the science platform
        session = Session()
        session_id = session.create(
            name  = ID,
            image = image,
            cores = cores,
            ram   = ram,
            kind  = "headless",
            cmd   = cmd,
            args  = f'{NAME} {ID}'
        )
    
        print("Sesion ID: {}".format(session_id[0]))
    
        # do not continue until bash script has completed
        t = 0
        while not os.path.exists(f'/arc/projects/salvage/ALMA_reduction/salvage_completion_files/{ID}_derived_complete.txt'):
            # check every minute
            time.sleep(60)
            t+=1
            print(f'Moment maps have not yet completed. Elapsed time: {t} min.')
    
    
        # wipe ALL data that is not the reduced image or derived products
        #os.system(f'rm -rf /arc/projects/salvage/ALMA_data/{ID}/')
    
        print('Galaxy reduction complete. Moving on to next galaxy.\n')

    else:

        print('Skipping STAGE 4: RUN PHANGS MOMENTS\n')
        




