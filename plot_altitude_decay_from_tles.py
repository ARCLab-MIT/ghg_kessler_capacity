'''
FILE DESCRIPTION: This script is used to plot the decay of a tracked object in the NORAD catalog 
over time. It reads in a history of TLEs, which are processed to compute the time-averaged altitude 
(as well as perigee and apogee altitudes) of the object over time. The script also plots the mass 
density of the atmosphere over time and altitude, as computed by NRLMSIS 2.0.  

TLEs for the tracked object of interest (in this case, SATCAT 4006) are stored in data/tles, and may be pulled
from space-track.org using the script pull_tles.py. 

The density background data is stored in data/dens_msis2.pkl, and is used to plot the mass density. 

10/10/2024 - William Parker
'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime as dt
import scipy.io
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
import matplotlib
from pymsis import msis
import pickle as pkl
import os
import pandas as pd

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['text.color'] = 'gray'
plt.rcParams['axes.labelcolor'] = 'gray'
plt.rcParams['xtick.color'] = 'gray'
plt.rcParams['ytick.color'] = 'gray'
plt.rcParams['axes.edgecolor'] = 'gray'

def main(): 
    # if density datafile already exists, load it. Otherwise, compute it and save it.
    dens_fname = 'data/dens_msis2.pkl' 
    if os.path.isfile(dens_fname):
        dbfile = open(dens_fname, 'rb')     
        dens = pickle.load(dbfile)
        tvec = pickle.load(dbfile)[1:]
        alt = pickle.load(dbfile)
    else:
        # if not already computed, just request compoute the density data
        tvec = dt.datetime(1968,1,1) + np.array([dt.timedelta(days = int(i)*30.4375) for i in range((2002-1968)*12)])
        alt = np.arange(200,1320,20)
        dens = compute_dens(tvec, alt)

    # compute tracked parameters from TLEs (TRUTH)
    norad = ['4006']
    fig, ax = plt.subplots(figsize = (3,7))
    im = ax.imshow(np.fliplr(dens).T, aspect = 'auto', cmap='Grays', interpolation = 'gaussian',extent=[min(tvec), max(tvec), min(alt),max(alt)], norm = matplotlib.colors.LogNorm())
    ax.xaxis_date()
    fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.07,label = 'Thermospere Mass Density $[\mathregular{kg/m^3}]$')

    # plot decay profile for each satelllite of interest
    for k in range(len(norad)): 
        path = 'data/' + norad[k] + '.txt'
        alt_sat,t_sat,v_sat, spec_energy, ecc, alt_per, alt_ap = altitudes_and_times_from_tles(path)

        alt_sat = np.flipud(alt_sat)
        alt_per = np.flipud(alt_per)
        alt_ap = np.flipud(alt_ap)
        t_sat = np.flipud(t_sat)
        v_sat = np.flipud(v_sat)
        ecc = np.flipud(ecc)

        # plot the tracked satellite
        plt.plot(t_sat,alt_sat, color = 'crimson',linestyle='solid', label = 'ID: '+str(norad[k])+', tracked', linewidth = 1.5)
        plt.fill_between(t_sat, alt_per, alt_ap, color = 'red', alpha = 0.1)
        plt.xlabel('Year') 
        plt.ylabel('Altitude [km]')
        plt.xlim([min(tvec), max(tvec)])
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        plt.ylim([200,1150])

    plt.tight_layout()
    plt.show()
 
def compute_dens(tvec, alt):
    # run NRLMSISE-2.0 for each year in f107_year_fit_proj with the alts from ALT -- sample across many lats and lons to get global average
    # get the multiplier for each year and altitude from NRLMSISE-00 compared to 2000, then compare to the RCP multipliers -- how off are they from where the natural deviation is?
    lats = np.arange(0,90,20)
    lons = np.arange(0,360,30)
    dens_p = np.zeros((len(lats), len(lons), len(alt), len(tvec)))
    dens = np.zeros((len(tvec),len(alt)))
    for i in range(len(tvec)):
        for m in range(len(alt)):
            for j in range(len(lats)):
                for k in range(len(lons)):
                    # query nrlmsise-00 for this combination of lat, lon, and alt
                    data = msis.run(tvec[i], float(lats[j]), float(lons[k]), float(alt[m]), geomagnetic_activity=-1, version=2)
                    dens_p[j,k,m,i] = data[0][0]

            # take average dens_p for each altitude each year
            dens[i,m] = np.average(dens_p[:,:,m,i])
        print(i/len(tvec))

        # save density data to pkl file
        with open('dens_msis2.pkl', 'wb') as f:
            pkl.dump(dens, f)
            pkl.dump(tvec, f)
            pkl.dump(alt, f)

    return dens

def get_tle_parameters(tle):
    norad = int(tle[0][2:7])
    epoch = float(tle[0][20:31]) 
    incl = float(tle[1][8:15])
    e = float('0.' + tle[1][26:32])
    mm = float(tle[1][52:62])
    T = 24*3600/mm
    mu = 398600
    re = 6378.15
    a = ((T/(2*3.14159))**2*mu)**(1/3) #Kepler's third law
    r_per = a*(1-e)
    r_ap = a*(1+e)
    alt_per = r_per-re
    alt_ap = r_ap-re
    v_avg = 2*np.pi*a/T# velocity of an equivalent circular orbit -- is this the same as the average velocity of the orbit?
    alt = a*(1+0.5*e**2)-re # see https://phys.libretexts.org/Bookshelves/Astronomy__Cosmology/Celestial_Mechanics_(Tatum)/09%3A_The_Two_Body_Problem_in_Two_Dimensions/9.10%3A_Mean_Distance_in_an_Elliptic_Orbit  #a-re#a*(1-e)-re
    delta = dt.timedelta(days = float(tle[0][20:31])-1)
    spec_energy = -mu/(2*a)
    if float(tle[0][18:20]) <  25:
        year = '20' + tle[0][18:20]
    else: 
        year = '19' + tle[0][18:20]
    t_dt = dt.date(int(year), 1, 1) + delta    
    tle_data = {'norad':norad,'epoch':epoch, 'incl':incl, 'e':e, 'mm':mm, 'T':T, 'mu':mu, 'a':a, 'v_avg':v_avg, 'alt':alt,'spec_energy':spec_energy, 't_dt':t_dt, 'alt_per':alt_per, 'alt_ap':alt_ap}
    return tle_data


def altitudes_and_times_from_tles(tle_file): 
    tle_list = []
    with open(tle_file, 'r') as file: 
        csvreader = csv.reader(file)
        for row in csvreader: 
            tle_list.append([row[0], next(file)])

    short_tles = np.zeros((len(tle_list), 11))
    i = 0
    timestamp = []
    for tle in tle_list:
        tle_params = get_tle_parameters(tle) 
        delta = dt.timedelta(days = float(tle[0][20:31])-1)
        if float(tle[0][18:20]) <  25:
            year = '20' + tle[0][18:20]
        else: 
            year = '19' + tle[0][18:20]
        timestamp.append(dt.date(int(year), 1, 1) + delta)

        short_tles[i,:] = [tle_params['norad'], tle_params['epoch'], tle_params['incl'], tle_params['e'], tle_params['mm'], tle_params['alt'], tle_params['a'], tle_params['v_avg'], tle_params['spec_energy'], tle_params['alt_per'], tle_params['alt_ap']]
        i += 1

    accum_altitudes = short_tles[:,5]
    accum_timestamp = timestamp
    v_avg_circ = short_tles[:,7]
    energy = short_tles[:,8]
    ecc = short_tles[:,3]
    alt_per = short_tles[:,9]
    alt_ap = short_tles[:,10]
    return accum_altitudes , accum_timestamp, v_avg_circ, energy, ecc, alt_per, alt_ap

if __name__ == '__main__': 
    main()