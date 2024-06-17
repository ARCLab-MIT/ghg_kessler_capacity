'''
FILE DESCRIPTION: This script is used to plot the decay of a tracked object in the NORAD catalog 
over time. It reads in a history of TLEs, which are processed to compute the time-averaged altitude 
(as well as perigee and apogee altitudes) of the object over time. The script also plots the mass 
density of the atmosphere over time and altitude, as recorded by Emmert (2021). 

TLEs for the tracked object of interest (in this case, SATCAT 4006) are stored in data/tles, and may be pulled
from space-track.org using the script pull_tles.py. 

The density background data is stored in data/emmert_cnossen_dens_full.mat, and is used to plot the mass density. 
The density of the background may also have been similarly modeled using NRLMSISE-00 and the known space weather 
drivers during this time. F10.7 and Ap are both plotted for reference during the time under study. 

05/01/2024 - William Parker
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

# set plot properties
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['text.color'] = 'gray'
plt.rcParams['axes.labelcolor'] = 'gray'
plt.rcParams['xtick.color'] = 'gray'
plt.rcParams['ytick.color'] = 'gray'
plt.rcParams['axes.edgecolor'] = 'gray'

def main(): 

    # load density model, from orbit-averaged densities from Emmert and Cnossen. Only the Emmert region of the density (1967-2019) is used in this figure.
    # Emmert: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JA029455
    # Cnossen: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022GL100693
    dens_model = scipy.io.loadmat('data/emmert_cnossen_dens_full.mat')
    dens = (dens_model['dens'])
    t_days = dens_model['t']
    alt = dens_model['alt']
    t_days = t_days[0]
    alt = alt[0]

    # convert t to datetime from 1/1/1967
    t = []
    for i in range(len(t_days)): 
        t.append(dt.date(1967,1,1) + dt.timedelta(days=int(t_days[i])))

    # compute tracked parameters from TLEs (measured)
    norad = ['4006']

    # plot density background
    max_alt = 1000
    min_alt = 250
    mint = np.where(t_days == (dt.datetime(1967,1,1)-dt.datetime(1967,1,1)).days)[0][0]
    maxt = np.where(t_days == (dt.datetime(2002,1,1)-dt.datetime(1967,1,1)).days)[0][0]
    maxh = np.where(alt==max_alt)[0][0]
    minh = np.where(alt==min_alt)[0][0]
    dens_fl = np.array((abs(dens[mint:maxt,:maxh])), dtype=float)

    x_min = mdates.date2num(t[mint])
    x_max = mdates.date2num(t[maxt])

    fig, ax = plt.subplots(dpi = 150)
    im = ax.imshow(np.fliplr(dens_fl).T, aspect = 'auto', cmap='Greys', extent=[x_min, x_max, min_alt,max_alt], norm = matplotlib.colors.LogNorm())
    ax.xaxis_date()
    fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.07, anchor=(0.5,0.0),label = 'Mass Density $[\mathregular{kg/m^3}]$')

    # for each satellite being tracked, compute plot the time-averaged altitude over time
    for k in range(len(norad)): 
        path = 'data/' + norad[k] + '.txt'
        alt_sat,t_sat,v_sat, spec_energy, ecc, alt_per, alt_ap = altitudes_and_times_from_tles(path)

        alt_sat = np.flipud(alt_sat)
        alt_per = np.flipud(alt_per)
        alt_ap = np.flipud(alt_ap)
        t_sat = np.flipud(t_sat)
        v_sat = np.flipud(v_sat)
        ecc = np.flipud(ecc)

        # plot altitude over time
        plt.plot(t_sat,alt_sat, color = 'k',linestyle='solid', label = 'ID: '+str(norad[k])+', tracked', linewidth = 1)
        plt.fill_between(t_sat, alt_per, alt_ap, color = 'tab:orange', alpha = 0.2)
        plt.xlabel('Year') 
        plt.ylabel('Altitude [km]')
        plt.xlim([mdates.date2num(dt.date(1967,1,1)), x_max])
        # set the label interval to be every 10 years
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        plt.ylim([250,1000])
    
    plt.tight_layout()
    plt.show()

    # Pull space weather data for reference!
    path = 'data/SW-All.mat'
    # use loadmat to get the data from the .mat file
    swdata = scipy.io.loadmat(path)
    # get the data from the dictionary
    data = swdata['sw_data']
    j = 0
    s_idx = 3381
    F107 = np.zeros((16165-s_idx,))
    Ap = np.zeros((16165-s_idx,))
    t = np.zeros((16165-s_idx,), dtype=dt.datetime)
    for i in np.arange(s_idx,16165):
        if data[i][25][0][0] < 60:
            F107[j] = F107[j-1]
        else:
            F107[j] = data[i][25][0][0]
        Ap[j] = data[i][20][0][0]
        t[j] = dt.datetime(1967,1,1) + dt.timedelta(days = j)
        j += 1

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t, F107, '-', linewidth = 0.5, color = 'darkcyan')
    plt.grid(axis = 'y', alpha = 0.2)
    plt.ylabel('F10.7 [sfu]')
    plt.xlim([dt.datetime(1967,1,1), dt.datetime(2002,1,1)])
    plt.ylim([60,400])
    plt.subplot(2,1,2)
    plt.plot(t, Ap, '-', linewidth = 0.5, color = 'sandybrown')
    plt.xlabel('Year')
    plt.ylabel('Ap [2nT]')
    plt.grid(axis = 'y', alpha = 0.2)
    plt.xlim([dt.datetime(1967,1,1), dt.datetime(2002,1,1)])
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()

    # make a plot with 2 y axes, one for F107 and the other for Ap, to show concurrently
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('F10.7 [sfu]', color=color)
    ax1.plot(t, F107, '-', linewidth = 0.5, color = color, alpha = 0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(axis = 'y', alpha = 0.2)
    ax1.set_xlim([dt.datetime(1967,1,1), dt.datetime(2002,1,1)])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Ap [2nT]', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, Ap, '-', linewidth = 0.5, color = color, alpha = 0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(axis = 'y', alpha = 0.2)
    ax2.set_xlim([dt.datetime(1967,1,1), dt.datetime(2002,1,1)])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def get_tle_parameters(tle):
    norad = int(tle[0][2:7])
    epoch = float(tle[0][20:31]) 
    incl = float(tle[1][8:15])
    e = float('0.' + tle[1][26:32])
    mm = float(tle[1][52:62])
    T = 23.9344*3600/mm
    mu = 398600
    re = 6378.15
    a = ((T/(2*3.14159))**2*mu)**(1/3) #Kepler's third law
    r_per = a*(1-e)
    r_ap = a*(1+e)
    alt_per = r_per-re
    alt_ap = r_ap-re
    v_avg = 2*np.pi*a/T# velocity of an equivalent circular orbit
    alt = a*(1+0.5*e**2)-re # see https://phys.libretexts.org/Bookshelves/Astronomy__Cosmology/Celestial_Mechanics_(Tatum)/09%3A_The_Two_Body_Problem_in_Two_Dimensions/9.10%3A_Mean_Distance_in_an_Elliptic_Orbit 
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
