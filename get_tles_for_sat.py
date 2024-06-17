'''
FILE DESCRIPTION: Define a satellite catalog (SATCAT) identifier of interest, and
download all of the available TLEs for that satellite from Space-Track.org. This script
will prompt the user for their Space-Track.org log-in credentials, and then download the
TLEs for the satellite of interest.  

05/01/2024 - William Parker
'''

import spacetrack.operators as op
from spacetrack import SpaceTrackClient
import datetime as dt
import numpy as np
from time import sleep
from pathlib import Path
from random import sample
import time

# Check whether ./Data/TLEs/ exists. If not, make it.
path = '../Data/TLEs/'
Path(path).mkdir(parents=True, exist_ok=True)

# Prompt user for Space-Track log-in credentials
print("Log in to using your Space-Track.org account credentials.\n")
st_email = input("Email: ")
st_pass = input("Password: ")

# Log in to Space-Track using your email and password
st = SpaceTrackClient(identity=st_email, password=st_pass)

# Define the satellite or satellites of interest according to their SATCAT identifier. 
# This can be a single satellite or a list of satellites. SATCAT identifiers for satellites of interest
# may be found at https://celestrak.org/satcat/search.php. 
satcats = [5784]


print("Define a study period (using yyyy-mm-dd formats).\n")
studyperiod_start = input("Start date: ")
studyperiod_end = input("End date: ")

# Only pull TLEs from between the start date and the end date
drange = op.inclusive_range(dt.datetime(int(studyperiod_start[0:4]), int(studyperiod_start[5:7]), int(studyperiod_start[8:10])), dt.datetime(int(studyperiod_end[0:4]), int(studyperiod_end[5:7]), int(studyperiod_end[8:10])))

# pull data, careful to not break the rules for data pulling from space-track. 
j = 1
t0 = time.time()
for satcat in satcats: 
    with open('../Data/ablestar_tles/'+str(satcat)+'.txt','w') as f: 
        f.write(st.tle(norad_cat_id=satcat, epoch=drange, orderby='epoch desc', format='tle'))
        f.close()
    print("Printed file: " + str(satcat) + ".txt.")
    t_now = time.time()
    t_left = (t_now-t0)/j*(len(satcats)-j)/3600
    print(str(j/len(satcats)*100)+'% complete, estimated time remaining is '+ str(t_left) +'h')
    for i in range(26):
        if i < 13:
            print("*"*(i+1))
            sleep(0.5)
        else:
            print("*"*(26-i))
            sleep(0.5)
    j += 1