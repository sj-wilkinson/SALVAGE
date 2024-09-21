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
    elif len(glob.glob(f'{PATH}/log/*casa-*.log'))>0:

        file_to_check = glob.glob(f'{PATH}/log/*casa-*.log')[0]

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

    ### if there are no casa logs, try using the casapy logs ###
    elif len(glob.glob(f'{PATH}/log/casapy*.log'))>0:

        file_to_check = glob.glob(f'{PATH}/log/casapy*.log')[0]

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
                method = 'casapy logs'


    ### include the case where logs have been zipped ###
    elif len(glob.glob(f'{PATH}/log/*.log.tgz'))>0:

        file_to_unzip = glob.glob(f'{PATH}/log/*.log.tgz')[0]

        print(f'CASA logs are zipped. Unzipping {file_to_unzip}')

        os.system(f'tar -xzf {file_to_unzip} -C {PATH}/log/')
        time.sleep(10)

        if len(glob.glob(f'{PATH}/log/*casa-*.log'))>0:

            file_to_check = glob.glob(f'{PATH}/log/*casa-*.log')[0]

        elif len(glob.glob(f'{PATH}/log/casapy*.log'))>0:

            file_to_check = glob.glob(f'{PATH}/log/casapy*.log')[0]

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
                method = 'zipped casa logs'

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
        #image='images.canfar.net/casa-6/casa:6.4.1-12-pipeline'
        image = 'images.canfar.net/casa-6/casa:6.5.4-9-pipeline'

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

    elif version == '4.7.1':
        print('WARNING: CASA v4.7.1 does not have a suggested version for restoring calibrations, using CASA v4.7.2 instead.')
        image='images.canfar.net/casa-4/casa:4.7.2'
        
    elif version == '4.7.0':
        image='images.canfar.net/casa-4/casa:4.7.2'
        #image='images.canfar.net/casa-4/casa:4.7.0'

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
        #print('WARNING: I am trialling using true CASA 4.2.0, no pipeline.')
        #image='images.canfar.net/casa-4/casa:4.2.0'
        
    else:
        print(f'Your CASA version ({version}) is either not supported or more recent than 6.1.1-15, defaulting to most recent: 6.5.4-9.')
        image = 'images.canfar.net/casa-6/casa:6.5.4-9-pipeline'

        # I keep receiving an error with 6.5.4-9: warnings.warn('PyFITS is deprecated, please use astropy.io.fits', let's try an earlier version?
        #print(f'Your CASA version ({version}) is either not supported or more recent than 6.1.1-15, defaulting to: 6.2.1-7.')
        #image = 'images.canfar.net/casa-6/casa:6.2.1-7-pipeline'   
        #image = "images.canfar.net/casa-6/casa:6.4.1-12-pipeline"

    return image


def get_path_to_data(ID, PROJ, MUID):

    '''
    Navigate through the directories within the project to get to the data.
    Knowing the path ahead of time is almost impossible to do reliably, so 
    just search starting from directory holding all data for target.

    In the newest iteration, I include the possibility of data being calibrated
    measurement sets from the ALMA Help Desk. In such a case, there is no complex
    directory structure, just the calibrated measurement set.
    '''

    # in some cases the MUID is given as a number, not a string, this was my work-around
    # I wonder now if this is no longer needed...
    try:
        UID =  MUID.split('/')[2] + '_' + MUID.split('/')[3] + '_' + MUID.split('/')[4]
    except:
        print('MUID is in the wrong format.')
        print(UID, type(MUID))
        return None, None

    # search all directories in the ALMA project folder for the relevant data
    PATH = None
    
    for root, dirs, files in os.walk(f"/arc/projects/salvage/ALMA_data/{ID}/"):

        # expected formatting for locally calibrated targets (with ScriptForPI.py)
        if "member.uid___"+UID in dirs:
            PATH = os.path.join(root, "member.uid___"+UID+'/calibrated/')

        # expected format for some ALMA Help Desk calibrated targets
        if "calibrated_final.ms" in dirs:
            PATH = f"/arc/projects/salvage/ALMA_data/{ID}/"

        # expected format for some ALMA Help Desk calibrated targets (2015/2016 batch)
        if "calibrated_final.ms_member.uid___."+UID in dirs:
            PATH = f"/arc/projects/salvage/ALMA_data/{ID}/"

    # if the data wasn't found, try checking if it is calibrated by the SRDP Service
    if PATH == None:

        SRDP = get_srdp_uid(MUID)

        for root, dirs, files in os.walk(f"/arc/projects/salvage/ALMA_data/downloads/"):

            # expected format for some ALMA Help Desk calibrated targets
            if SRDP in dirs:
                PATH = f"/arc/projects/salvage/ALMA_data/downloads/{SRDP}/"
    
    if PATH == None:
        print()
        print(f'Path to data not found ({ID}).\n')
        print()

    return PATH, UID


def get_srdp_uid(muid):

    if muid == 'uid://A001/X2f52/X2cf':
        srdp_uid = 'uid___A002_X10275c0_X42f2'

    if muid == 'uid://A001/X3578/Xe1':
        srdp_uid = 'uid___A002_X111f25c_X4a6'

    if muid == 'uid://A001/X2f52/X2ff':
        srdp_uid = 'uid___A002_X103d73c_X663'

    if muid == 'uid://A001/X2f52/X2e6':
        srdp_uid = 'uid___A002_X102cd30_Xaf68'

    if muid == 'uid://A001/X133d/X4226':
        srdp_uid = 'uid___A002_Xda64e4_X1673'

    if muid == 'uid://A001/X15a2/X72b':
        srdp_uid = 'uid___A002_Xf61d3a_X11e1'

    if muid == 'uid://A001/X2f52/X2fc':
        srdp_uid = 'uid___A002_X102cd30_X3182'

    return srdp_uid
    

###########################################

#### STAGE 0: READ IN SAMPLE TO REDUCE ####

###########################################

fpath = '/arc/projects/salvage/ALMA_reduction/samples/'
file  = 'salvage-AGN-Aug26-sample_match-lt-4rp_mrs-gt-2rp_qa2-pass_no-inf-dups.txt'
file  = 'salvage-AGN-Aug29-sample_match-lt-4rp_mrs-gt-1p5rp_res-lt-3p05_qa2-pass_no-inf-dups.txt'
file  = 'salvage-AGN-Sep20-sample_match-lt-4rp_mrs-gt-1p5rp_res-lt-3p05_qa2-pass_no-inf-dups.txt'

objID_sample, year_sample, muid_sample, auid_sample, proj_sample, name_sample = np.loadtxt(fpath+file, unpack = True, dtype = str, usecols = [0,9,10,11,12,13])
z_sample, mass_sample, rpetro_sample, ra_sample, dec_sample, res_sample, mrs_sample, sens_sample = np.loadtxt(fpath+file, unpack = True, dtype = float, usecols = [1,2,3,4,5,6,7,8])

# galaxies to reduce in this run are given by inputs from the user
argv = sys.argv
min_index = int(argv[-2])
max_index = int(argv[-1])

#rerun_targets = ['587727944034091201'] # the rest of 2018.1.01852.S is complete, try this again
#rerun_targets = ['587729776375562308'] # the only target from 2016.1.01269.S, try this again (I think I messed this one up, try again later?)
#rerun_targets = ['587726100411121941'] # files not where script expects, copied and re-running...
rerun_targets = ['588017703458111630', '587732772108042456', '587726015069421743'] # 2015 and 2016 project codes with previously successful targets, these should work, right? It would be silly to send them off
rerun_targets = ['588848899929080012', '587736543088672829'] # 2017 project codes with previously successful targets
rerun_targets = ['587742774013657229'] # 2021.1.01089.S (Cortese) the final one
#rerun_targets = ['587732772108042456'] # sent to Help desk as a problem, but has been imaged successfully in the meantime... run moment maps? # deleted directory by accident (1.3 TB it was taking up, so replace with Help Desk calibrated MS for future imaging.)

rerun_targets = ['587739651034054894', '587741533326868609'] # calibrated by help desk but never actually tried myself...? No directories in ALMA_data/, # maybe 587727220865040712?

# maybe I can put these in a text file and read them in so its not so ugly?
help_desk_targets = [# Cycle 1/2 targets sent
                     '587725073918722287', '587725073918722357', '587725074455724392', '587726015085478016', \
                     '587726032227336433', '587726032228188446', '587726032228909342', '587726032234610739', \
                     '587726032764076412', '587726032766828783', '587726033300881540', '587726033335550094', \
                     '587727942952550837', '587727942952681722', '587727943490273729', '587727944027996390', \
                     '587727944562573646', '587727944563687568', '587728879256207619', '587734892755353685', \
                     '587736541485400245', '587736547388031357', '587736803470540958', '587736975275327817', \
                     '587736976347693299', '587738568707604532', '587739406766833786', '587739646199201823', \
                     '587739647273533569', '587742575921594439', '587742627987980483', '587742628524720281', \
                     '588015509267284063', '588017704006058105', '588848899357147414', '588848899895853247', \
                     '588848900429185386', '588848900429185433', '588848900966449330', \
                    # 2015/2016 targets sent
                     '588017703458111630', '587726100411121941', '587726102556180606', '587741533326868609', \
                     '587739651034054894', '587727220865040712', '588848900966514994', '587732772108042456', \
                     '587726015069421743','587730817902116911', '587726877264249088', '587729776375562308']

srdp_targets = ['587727944034091201', '587742774013657229', '587734621630431417', '587735236883513572', \
                '587742551761027182', '587739845391876411', '587742615097442546', '588023669708816634', \
                '587742062148911209', '587741724972548183', '587741602035138654', '587739407875113205', \
                '587736940914409584']

rerun_targets = srdp_targets.copy()

do_stage1 = False
do_stage2 = False
do_stage3 = True
do_stage4 = True

rerun_only = True
skip_completed = True
skip_help_desk_targets = False
skip_srdp_targets = False
wipe_downloads = True

# loop over galaxies and launch jobs
for i in np.arange(min_index,max_index):

    ID   = objID_sample[i]
    NAME = name_sample[i]
    MUID = muid_sample[i]
    PROJ = proj_sample[i]
    Z    = z_sample[i]
    DIST = cosmo.angular_diameter_distance(Z).value
    RA   = ra_sample[i]
    DEC  = dec_sample[i]
    YEAR = year_sample[i]

    # choose to only run on problem galaxies from a previous run
    if (ID not in rerun_targets) & rerun_only:
        continue

    # skip if SDSS object already has a PHANGS moment 0 map
    if os.path.exists(f'/arc/projects/salvage/ALMA_reduction/phangs_pipeline/derived/{ID}/{ID}_12m_co10_strict_mom0.fits') & skip_completed:
        
        print('##############################################################################')
        print(f'Skipping {ID} because it already has a moment 0 map from the PHANGS pipeline.')
        print('##############################################################################')

                            # don't wipe the calibrated MS's supplied by the help desk!
        if wipe_downloads & (ID not in help_desk_targets) & (ID not in srdp_targets):
        
            print('Wiping downloads, accordingly.')
            print(f'rm -rf /arc/projects/salvage/ALMA_data/{ID}/*.tar')
            os.system(f'rm -rf /arc/projects/salvage/ALMA_data/{ID}/*.tar')
            os.system(f'rm -rf /arc/projects/salvage/ALMA_data/{ID}/*.pickle')

            PATH, UID = get_path_to_data(ID, PROJ, MUID)

            if PATH != None:

                os.system(f'rm -rf {PATH}/raw/*')
                os.system(f'rm -rf {PATH}/calibrated/working/*')

        continue

    if skip_help_desk_targets & (ID in help_desk_targets):

        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print(f'Skipping {ID} because was calibrated by the ALMA Help Desk.')
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

        continue

    if skip_srdp_targets & (ID in srdp_targets):

        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(f'Skipping {ID} because was calibrated by the SRDP Service.')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        continue


    # allow early cycles to run through the pipeline, but make sure download and calibration is skipped...
    if (ID in help_desk_targets) or (ID in srdp_targets):

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'{ID} was calibrated by the ALMA Help Desk or the SRDP service. Carry on.')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print()

        do_stage1 = False
        do_stage2 = False
        

    # remove completion flag files for this galaxy
    os.system(f'rm -rf /arc/projects/salvage/ALMA_reduction/salvage_completion_files/{ID}_*_complete.txt')

    #######################################
    
    ##### STAGE 1: DOWNLOAD AND UNZIP #####

    #######################################

    # select appropriate resources
    image = "images.canfar.net/skaha/astroml:24.03"
    cmd = '/arc/projects/salvage/ALMA_reduction/bash_scripts/download_and_unzip_revert.sh'
    ram=2
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
    #ram=12
    ram=8
    cores=2

    # identify file path to data
    PATH, UID = get_path_to_data(ID, PROJ, MUID)
    
    if PATH == None:
        print()
        print(f'Path to data not found. Skipping this galaxy ({ID}).\n')
        print()
        #continue

    if do_stage2:

        # identify and select the appropriate CASA version
        version, calib, method = get_casa_version(PATH.replace('/calibrated/', ''), UID)
        image = casa_version_to_canfar_image(version)
        print(ID, version, calib, method, image)
            
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
    
    ms_key_out = f'ms_file_key_salvage_{ID}.txt'
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

    # prep file path for PHANGS-ALMA pipeline (depending on where the data was calibrated ...)
    ms_root = '/arc/projects/salvage/ALMA_data/' # root is always the same
    ms_filepath = PATH.replace(ms_root, '') # wipe ms_root directory so it can be added separately

    # take all measurement set files from this directory and let PHANGS deal with them
    # could be multiple observations to meet QA and/or duplicate but approx equally good observations
    ms_files = glob.glob(ms_root + ms_filepath + '*.ms*')

    print(ms_root + ms_filepath, os.path.isdir(ms_root + ms_filepath), len(ms_files))

    if len(ms_files)<1:
        #checking if files are missing because of bug in previous stages or this one...
        print('No MS files found in ', ms_filepath)

        print('CANCEL IMAGING DUE TO MISSING FILES.\n')
        continue

    # remove .flagversions (and .ampli_inf) -- they seem superfluous and causing trouble
    ms_files_keep = []
    for ms_file_ in ms_files:
        if ms_file_[-13:] == '.flagversions':
            continue
        elif ms_file_[-10:] == '.ampli_inf':
            continue
        else:
            ms_files_keep.append(ms_file_.replace(ms_root, ''))

    print()
    print('The following ms files were in the above directory.')
    print(ms_files)
    print('After removing those with .flagversions (which seem to cause trouble), the following are kept.')
    print(ms_files_keep)
    print()

    ms_files = ms_files_keep.copy()

    # check if there are visibilities for this target from another project and add on top if there is...
    #init = len(np.array(objID_in_file)[np.array(objID_in_file)==ID])
    init = 0

    # add each measurement set to ms key
    for j in range(init, init + len(ms_files)):

        ms_path = ms_files[j-init].replace(ms_root, '') # wipe ms_root directory so it can be added separately
        obs_num = j+1 # record different measurement sets as different observations

        out_str += f"{ID} {ID}_{obs_num} all 12m {obs_num} {ms_path}\n"
        #out_str += f"{ID} {ID}_{obs_num} all 12m {obs_num} {ms_path}.split\n" 
        # need to accomodate other observations of the same objID?

    # replace placeholder "__DATA__" in template with formatted data
    out_data = out_data.replace('__DATA__', out_str)
        
    # write to new key file
    out = open(key_dir + ms_key_out, 'w')
    out.write(out_data)
    out.close()

    ## write function to make distance key for single target

    dist_key_out = f'distance_key_salvage_{ID}.txt'
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

    targ_key_out = f'target_definitions_salvage_{ID}.txt'
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
    velwidth = 1200

    out_str += f"{ID} {coord_str} {sys_vel} {velwidth}\n" 
    # need to accomodate other observations of the same objID?
    
    # replace __DATA__ in template with formatted data
    out_data = out_data.replace('__DATA__', out_str)
    
    # write to new key file
    out = open(key_dir + targ_key_out, 'w')
    out.write(out_data)
    out.close()

    ## generate master key file that points to these keys

    master_key_out = f'master_key_salvage_{ID}.txt'
    master_key_tmp = 'master_key_template.txt'

    # get template master key
    out = open(key_dir + master_key_tmp, 'r')
    out_data = out.read()
    out.close()
    
    # replace __OBJID__ in template with the SDSS objID for this galaxy
    out_data = out_data.replace('__OBJID__', ID)

    # write to new key file
    out = open(key_dir + master_key_out, 'w')
    out.write(out_data)
    out.close()
    

    #########################################
    
    ##### STAGE 3b: RUN PHANGS PIPELINE #####

    #########################################

    if do_stage3:

        # select appropriate resources
        #image = "images.canfar.net/casa-6/casa:6.5.6-22"
        #image = "images.canfar.net/casa-6/casa:6.5.4-9-pipeline"
        image = "images.canfar.net/casa-6/casa:6.4.1-12-pipeline"
        cmd = '/arc/projects/salvage/ALMA_reduction/bash_scripts/run_PHANGS_pipeline.sh'
        ram=32
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
            args  = f'{NAME} {ID} {key_dir + ms_key_out}'
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
        image = "images.canfar.net/skaha/astroml:24.03"
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
    
        print('Galaxy reduction complete. Moving on to next galaxy.\n')

    else:

        print('Skipping STAGE 4: RUN PHANGS MOMENTS\n')


    # if moment maps were generated, wipe superfluous data
    if os.path.exists(f'/arc/projects/salvage/ALMA_reduction/phangs_pipeline/derived/{ID}/{ID}_12m_co10_strict_mom0.fits') & wipe_downloads:
        
        print('#######################################################################')
        print(f'{ID} was successful, deleting superfluous data.')
        print('#######################################################################')

        os.system(f'rm -rf /arc/projects/salvage/ALMA_data/{ID}/*.tar')
        os.system(f'rm -rf /arc/projects/salvage/ALMA_data/{ID}/*.pickle')
        os.system(f'rm -rf {PATH}/raw/*')
        os.system(f'rm -rf {PATH}/product/*')
        os.system(f'rm -rf {PATH}/calibrated/working/*')
        



