import os, alminer, pandas

environ = os.environ
ID = str(environ['ID'])
NAME = str(environ['NAME'])
PROJ = str(environ['PROJ'])

out_dir = f'/arc/projects/salvage/ALMA_data/{ID}/'

#print('Checking ALMA archive for target...')
#myquery = alminer.keysearch({'proposal_id': [PROJ], 'target_name':[NAME]})

print('Checking ALMA archive for targets in given proposal...')
myquery = alminer.keysearch({'proposal_id': [PROJ]})

print('Downloading with ALMiner...')
alminer.download_data(myquery[myquery['ALMA_source_name']==NAME], fitsonly=False, dryrun=False, location=out_dir, print_urls=False, archive_mirror='NRAO')

print('Python script complete.')
