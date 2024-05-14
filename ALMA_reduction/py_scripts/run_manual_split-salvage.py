import os
import numpy as np
from casatasks import split

NAME  = os.environ["NAME"]
OBJID = os.environ["OBJID"]
MSFILE  = os.environ["MSFILE"]

key_dir = '/arc/projects/salvage/phangs_imaging_scripts/phangs-alma_keys/'
data_dir = '/arc/projects/salvage/ALMA_data/'

vis_list = np.array(np.loadtxt(MSFILE, unpack = True, usecols = [5], dtype = str), ndmin = 1)

for vis in vis_list:

    print(f'Splitting out {NAME} from {vis}.')

    #split(vis=data_dir+vis, intent='OBSERVE_TARGET*', field=NAME, spw='', datacolumn='CORRECTED', outputvis=data_dir+vis, keepflags=False, timebin='0s')

    # split out data, must delete existing output; split will not overwrite
    vis_out = vis+'.split'
    os.system(f'rm -rf {data_dir+vis_out}')
    split(vis=data_dir+vis, intent='OBSERVE_TARGET*', field=NAME, spw='', datacolumn='all', outputvis=data_dir+vis_out, keepflags=False, timebin='0s')

    # use split to copy data back into its original name?
    #os.system(f'rm -rf {vis_out}')
    #split(vis=data_dir+vis_out, field=NAME, spw='', datacolumn='', outputvis=data_dir+vis, keepflags=False, timebin='0s')

    # remove temporary split?
    #os.system(f'rm -rf {vis_out}')

    # export output_vis so it can be deleted after PHANGS-ALMA pipeline
    os.system(f'export output_vis={data_dir+vis_out}')

## UPDATE MS KEY FILE

obs_num_list = np.array(np.loadtxt(MSFILE, unpack = True, usecols = [4], dtype = str), ndmin = 1)

ms_key_tmp = 'ms_file_key_template.txt'

# get template table (formatting with no file paths)
out = open(key_dir + ms_key_tmp, 'r')
out_data = out.read()
out.close()

# wipe old content from output file
out = open(MSFILE, 'w')
out.close()

# initialize string
out_str = ''

for vis, obs_num in zip(vis_list, obs_num_list):

    out_str += f"{OBJID} {OBJID}_{obs_num} all 12m {obs_num} {vis}.split\n"

# replace placeholder "__DATA__" in template with formatted data
    out_data = out_data.replace('__DATA__', out_str)
        
    # write to new key file
    out = open(MSFILE, 'w')
    out.write(out_data)
    out.close()

print('Split complete.')