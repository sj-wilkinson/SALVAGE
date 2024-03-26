import os, alminer, pandas

environ = os.environ
NAME = str(environ['NAME'])
ID = str(environ['ID'])

# loading ALMiner query object from the crossmatch between SDSS and the ALMA archive
f = '/arc/projects/salvage/ALMAxmatch/ALminer_output/SDSS-ALMA-search-Jan15-total.pkl'
myquery = pandas.read_pickle(f)

print('... Downloading with ALMiner.')
alminer.download_data(myquery[myquery['ALMA_source_name']==NAME], fitsonly=False, dryrun=False, location=f'/arc/projects/salvage/ALMA_data/{ID}/{NAME}/', print_urls=False, archive_mirror='NRAO')

print('Python script complete.')










#myquery = alminer.conesearch(ra=RA, dec=DEC, point=False, search_radius=0.00001) #several targets at RA and DEC
#myquery = alminer.target(NAME) #alma target name not in SIMBAD
#myquery = alminer.keysearch({'target_name': NAME})

#print('... Conducting ALMiner search.')
#myquery = alminer.conesearch(ra=RA, dec=DEC, point=True)