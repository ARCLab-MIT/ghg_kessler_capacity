'''
FILE DESCRIPTION: This script reads in the NetCDF file containing the atmospheric mass density
forecasts as were computed in Brown et al (2024) and stored here: https://edata.bham.ac.uk/1075/.
Future density reductions (in multipliers compared to the year 2000) are computed according to the 
shared socioeconomic pathways (SSPs) and assuming solar cycles similar to the past sixty years. A
sinusoid is fit to the historical record of F10.7 to generate a projected solar cycle.  
A slight extrapolation beyond the modeled upper bound of CO2 concentration is necessary for the years
2080-2100 for the SSP-8.5 case. 

The output of this script is stored in data/dens_forecast_ssp.mat, which contains the density profiles 
of interest for the rest of the paper.

05/01/2024 - William Parker
'''

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
import scipy.io
import datetime as dt
import pandas as pd
from nrlmsise00 import msise_model
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

# make arial font
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
# make font gray
plt.rcParams['text.color'] = 'gray'
plt.rcParams['axes.labelcolor'] = 'gray'
plt.rcParams['xtick.color'] = 'gray'
plt.rcParams['ytick.color'] = 'gray'
# make the box around the plot gray
plt.rcParams['axes.edgecolor'] = 'gray'

file = "data/DEN-CO2scaling2000-2500_v2.nc"

#%% 
def main(): 
    
    processed_file = 'data/dens_forecast_sspX.mat'
    
    if not os.path.exists(processed_file):
        # Unload variables
        data = nc.Dataset(file, 'r', Format='NETCDF4')
        alt = data.variables['alt'][:]
        co2 = data.variables['CO2'][:]
        f107 = data.variables['F107'][:]
        year = data.variables['year'][:]
        co2_only = data.variables['CO2_only'][:]
        # alt(9), year(101), f107(5)
        rcp30  = data.variables['RCP30'][:]
        rcp45 = data.variables['RCP45'][:]
        rcp60 = data.variables['RCP60'][:]
        rcp85 = data.variables['RCP85'][:]
        ssp1_19 = data.variables['SSP1_19'][:]
        ssp1_26 = data.variables['SSP1_26'][:]
        ssp2_45 = data.variables['SSP2_45'][:]
        ssp3_70 = data.variables['SSP3_70'][:]
        ssp3_70_lowNTCF = data.variables['SSP3_70_lowNTCF'][:]
        ssp4_34 = data.variables['SSP4_34'][:]
        ssp4_60 = data.variables['SSP4_60'][:]
        ssp5_34_over = data.variables['SSP5_34_over'][:]
        ssp5_85 = data.variables['SSP5_85'][:]

        # get time-dependent F10.7 to query the SSPs (use known F10.7 from 2000-2024, then projected forward)
        date_hist, f107_hist = project_f107()  
        
        # convert date_hist from datetime object to years since 2000
        date_hist_year = np.array([(date_hist[i] - dt.datetime(2000,1,1)).days for i in range(len(date_hist))])/365.25 
        
        # compute nominal density for every combination of f107 and altitude
        dens = compute_dens(f107, alt)
        
        # re-sample interpolate
        year_rs = np.arange(0,100 + 1/12, 1/12)+2000 
        f107_cs = scipy.interpolate.CubicSpline(date_hist_year+2000, f107_hist, )
        f107_rs = f107_cs(year_rs)  
        
        # plt.figure()
        # plt.semilogy(year_rs, f107_rs)
        # plt.show()  
        
        # if any value exceeds the maximum value of f107 in the input, set it to the maximum value
        f107_rs = np.array([f107_rs[i] if f107_rs[i] < np.max(f107) else np.max(f107) for i in range(len(f107_rs))])
        # repeat for minimum
        f107_rs = np.array([f107_rs[i] if f107_rs[i] > np.min(f107) else np.min(f107) for i in range(len(f107_rs))])
        alt_rs = np.arange(200,1000+5, 5)

        # plt.figure()
        # plt.plot(year_rs, f107_rs, 'r-')
        # plt.xlabel('Year')
        # plt.ylabel('F10.7 [sfu]')
        # plt.grid(axis = 'both', color = 'whitesmoke')
        # plt.show()
        
        # Assuming alt, year, f107 are 1D arrays defining a 3D grid
        alt_grid, year_grid, f107_grid = np.meshgrid(alt, year, f107, indexing='ij')
        f107_dens_grid , alt_dens_grid= np.meshgrid(f107, alt, indexing='ij')
        
        # plot ssp5_85 and see if reasonable extrapolation can be made
        # plt.figure()
        # for i in range(len(alt)):
        #     plt.plot(year , ssp5_85[i,:,4].T)
        # plt.show()
        
        # perform linear extrapolation of ssp5_85 to 2100 
        for i in range(len(alt)):
            for j in range(len(year)):
                for k in range(len(f107)):
                    if np.ma.is_masked(ssp5_85[i,j,k]):
                        ssp5_85[i,j,k] = ssp5_85[i,j-1,k] + (ssp5_85[i,j-1,k] - ssp5_85[i,j-2,k])
        
        # plot ssp5_85_extrap and see if reasonable extrapolation can be made
        # plt.figure()
        # for i in range(len(alt)):
        #     plt.plot(year , ssp5_85[i,:,0].T)
        # plt.show()
            
        
        # Create the interpolator
        interp_ssp1_19 = RegularGridInterpolator((alt, year, f107), np.array(ssp1_19))
        interp_ssp1_26 = RegularGridInterpolator((alt, year, f107), np.array(ssp1_26))
        interp_ssp2_45 = RegularGridInterpolator((alt, year, f107), np.array(ssp2_45))
        interp_ssp3_70 = RegularGridInterpolator((alt, year, f107), np.array(ssp3_70))
        interp_ssp3_70_lowNTCF = RegularGridInterpolator((alt, year, f107), np.array(ssp3_70_lowNTCF))
        interp_ssp4_34 = RegularGridInterpolator((alt, year, f107), np.array(ssp4_34))
        interp_ssp4_60 = RegularGridInterpolator((alt, year, f107), np.array(ssp4_60))
        interp_ssp5_34_over = RegularGridInterpolator((alt, year, f107), np.array(ssp5_34_over))
        interp_ssp5_85 = RegularGridInterpolator((alt, year, f107), np.array(ssp5_85))
        
        interp_dens = RegularGridInterpolator((f107, alt), np.log(dens))
        # interp_dens = scipy.interpolate.interp2d(f107_dens_grid,alt_dens_grid ,np.log(dens),  kind='linear')
        
        # plt.figure()
        dens_vec = np.zeros(len(alt_rs))
        for i in range(len(alt_rs)):
            dens_vec[i] = interp_dens([f107[0], alt_rs[i]])
        dens_vec_real = np.zeros(len(f107))
        for j in range(len(f107)):
            dens_vec_real[j] = interp_dens([f107[j], alt[5]])
        
        
        # plt.semilogy(alt_rs, np.exp(dens_vec), 'r.')
        # plt.semilogy(f107, np.exp(dens_vec_real), 'r')
        # plt.show()
        
        dens_ssp1_19_rs = np.zeros((len(year_rs), len(alt_rs)))
        dens_ssp1_26_rs = np.zeros((len(year_rs), len(alt_rs)))
        dens_ssp2_45_rs = np.zeros((len(year_rs), len(alt_rs)))
        dens_ssp3_70_rs = np.zeros((len(year_rs), len(alt_rs)))
        dens_ssp3_70_lowNTCF_rs = np.zeros((len(year_rs), len(alt_rs)))
        dens_ssp4_34_rs = np.zeros((len(year_rs), len(alt_rs)))
        dens_ssp4_60_rs = np.zeros((len(year_rs), len(alt_rs)))
        dens_ssp5_34_over_rs = np.zeros((len(year_rs), len(alt_rs)))
        dens_ssp5_85_rs = np.zeros((len(year_rs), len(alt_rs)))
        dens_rs = np.zeros((len(year_rs), len(alt_rs)))
        for j in range(len(year_rs)):
            for i in range(len(alt_rs)):
                ssp1_19_rs = interp_ssp1_19([alt_rs[i], year_rs[j], f107_rs[j]])
                ssp1_26_rs = interp_ssp1_26([alt_rs[i], year_rs[j], f107_rs[j]])
                ssp2_45_rs = interp_ssp2_45([alt_rs[i], year_rs[j], f107_rs[j]])
                ssp3_70_rs = interp_ssp3_70([alt_rs[i], year_rs[j], f107_rs[j]])
                ssp3_70_lowNTCF_rs = interp_ssp3_70_lowNTCF([alt_rs[i], year_rs[j], f107_rs[j]])
                ssp4_34_rs = interp_ssp4_34([alt_rs[i], year_rs[j], f107_rs[j]])
                ssp4_60_rs = interp_ssp4_60([alt_rs[i], year_rs[j], f107_rs[j]])
                ssp5_34_over_rs = interp_ssp5_34_over([alt_rs[i], year_rs[j], f107_rs[j]])
                ssp5_85_rs = interp_ssp5_85([alt_rs[i], year_rs[j], f107_rs[j]])
                
                dens_rs[j,i] = np.exp(interp_dens([f107_rs[j], alt_rs[i]]))
                dens_ssp1_19_rs[j,i] = ssp1_19_rs*dens_rs[j,i]
                dens_ssp1_26_rs[j,i] = ssp1_26_rs*dens_rs[j,i]
                dens_ssp2_45_rs[j,i] = ssp2_45_rs*dens_rs[j,i]
                dens_ssp3_70_rs[j,i] = ssp3_70_rs*dens_rs[j,i]
                dens_ssp3_70_lowNTCF_rs[j,i] = ssp3_70_lowNTCF_rs*dens_rs[j,i]
                dens_ssp4_34_rs[j,i] = ssp4_34_rs*dens_rs[j,i]
                dens_ssp4_60_rs[j,i] = ssp4_60_rs*dens_rs[j,i]
                dens_ssp5_34_over_rs[j,i] = ssp5_34_over_rs*dens_rs[j,i]
                dens_ssp5_85_rs[j,i] = ssp5_85_rs*dens_rs[j,i]
                
                print(j/len(year_rs))
        
        # now we should have density as a function of time and altitude for each case
        # save each profile to a mat file
        scipy.io.savemat('data/dens_forecast_ssp.mat', mdict={'dens_ssp1_19_rs': dens_ssp1_19_rs, 'dens_ssp1_26_rs': dens_ssp1_26_rs, 'dens_ssp2_45_rs': dens_ssp2_45_rs, 'dens_ssp3_70_rs': dens_ssp3_70_rs, 'dens_ssp3_70_lowNTCF_rs': dens_ssp3_70_lowNTCF_rs, 'dens_ssp4_34_rs': dens_ssp4_34_rs, 'dens_ssp4_60_rs': dens_ssp4_60_rs, 'dens_ssp5_34_over_rs': dens_ssp5_34_over_rs, 'dens_ssp5_85_rs': dens_ssp5_85_rs, 'dens_rs':dens_rs, 'alt_rs': alt_rs, 'year_rs': year_rs})
    
    else: 
        plot_profiles(processed_file)
        
def plot_profiles(processed_file):
    data = scipy.io.loadmat(processed_file)
    # dens_ssp1_19_rs = data['dens_ssp1_19_rs']
    dens_ssp1_26_rs = data['dens_ssp1_26_rs']
    dens_ssp2_45_rs = data['dens_ssp2_45_rs']
    dens_ssp3_70_rs = data['dens_ssp3_70_rs']
    # dens_ssp3_70_lowNTCF_rs = data['dens_ssp3_70_lowNTCF_rs']
    # dens_ssp4_34_rs = data['dens_ssp4_34_rs']
    # dens_ssp4_60_rs = data['dens_ssp4_60_rs']
    # dens_ssp5_34_over_rs = data['dens_ssp5_34_over_rs']
    dens_ssp5_85_rs = data['dens_ssp5_85_rs']
    alt_rs = data['alt_rs']
    year_rs = data['year_rs']
    dens_rs = data['dens_rs']
    
    # make a figure that plots the atmospheric density over time for each altitude for each SSP
    # plt.figure()
    # plt.plot(year_rs.T, dens_ssp1_26_rs[:,np.argmin(np.abs(400-alt_rs))], '#118AB2', label = 'SSP1-26')
    # plt.plot(year_rs.T, dens_ssp2_45_rs[:,np.argmin(np.abs(400-alt_rs))], '#06D6A0', label = 'SSP2-45')
    # plt.plot(year_rs.T, dens_ssp3_70_rs[:,np.argmin(np.abs(400-alt_rs))], '#FFD166', label = 'SSP3-70')
    # plt.plot(year_rs.T, dens_ssp5_85_rs[:,np.argmin(np.abs(400-alt_rs))], '#EF476F', label = 'SSP5-85')
    # plt.plot(year_rs.T, dens_rs[:,np.argmin(np.abs(400-alt_rs))], '#073B4C', label = 'Baseline')
    # plt.xlabel('Year')
    # plt.ylabel('Atmospheric Density [kg/m^3]')
    # plt.grid(axis = 'both', color = 'gainsboro')
    # plt.legend()
    # plt.show()
    
    
    # plot dens_ssp/dens_rs for each ssp and show cases at 400, 600, and 800 km
    
    # plt.figure()
    # plt.subplot(1,3,1)
    # alt_plt = 200
    # plt.plot(year_rs.T, dens_ssp1_26_rs[:,np.argmin(np.abs(alt_plt-alt_rs))]/dens_rs[:,np.argmin(np.abs(alt_plt-alt_rs))], '#118AB2', label = 'SSP1-26')
    # plt.plot(year_rs.T, dens_ssp2_45_rs[:,np.argmin(np.abs(alt_plt-alt_rs))]/dens_rs[:,np.argmin(np.abs(alt_plt-alt_rs))], '#06D6A0', label = 'SSP2-45')
    # plt.plot(year_rs.T, dens_ssp3_70_rs[:,np.argmin(np.abs(alt_plt-alt_rs))]/dens_rs[:,np.argmin(np.abs(alt_plt-alt_rs))], '#FFD166', label = 'SSP3-70')
    # plt.plot(year_rs.T, dens_ssp5_85_rs[:,np.argmin(np.abs(alt_plt-alt_rs))]/dens_rs[:,np.argmin(np.abs(alt_plt-alt_rs))], '#EF476F', label = 'SSP5-85')
    # plt.ylim([0,1])
    # plt.xlabel('Year')
    # plt.title('200 km')
    # plt.ylabel('Density Multitplier')
    # plt.grid(axis = 'both', color = 'gainsboro', linewidth = 0.5)
    # plt.legend()
    
    # plt.subplot(1,3,2)
    # alt_plt = 500
    # plt.plot(year_rs.T, dens_ssp1_26_rs[:,np.argmin(np.abs(alt_plt-alt_rs))]/dens_rs[:,np.argmin(np.abs(alt_plt-alt_rs))], '#118AB2', label = 'SSP1-26')
    # plt.plot(year_rs.T, dens_ssp2_45_rs[:,np.argmin(np.abs(alt_plt-alt_rs))]/dens_rs[:,np.argmin(np.abs(alt_plt-alt_rs))], '#06D6A0', label = 'SSP2-45')
    # plt.plot(year_rs.T, dens_ssp3_70_rs[:,np.argmin(np.abs(alt_plt-alt_rs))]/dens_rs[:,np.argmin(np.abs(alt_plt-alt_rs))], '#FFD166', label = 'SSP3-70')
    # plt.plot(year_rs.T, dens_ssp5_85_rs[:,np.argmin(np.abs(alt_plt-alt_rs))]/dens_rs[:,np.argmin(np.abs(alt_plt-alt_rs))], '#EF476F', label = 'SSP5-85')
    # plt.xlabel('Year')
    # plt.title('500 km')
    # plt.ylim([0,1])
    # # plt.ylabel('Density [kg/m^3]')
    # plt.grid(axis = 'both', color = 'gainsboro', linewidth = 0.5)
    
    # plt.subplot(1,3,3)
    # alt_plt = 900
    # plt.plot(year_rs.T, dens_ssp1_26_rs[:,np.argmin(np.abs(alt_plt-alt_rs))]/dens_rs[:,np.argmin(np.abs(alt_plt-alt_rs))], '#118AB2', label = 'SSP1-26')
    # plt.plot(year_rs.T, dens_ssp2_45_rs[:,np.argmin(np.abs(alt_plt-alt_rs))]/dens_rs[:,np.argmin(np.abs(alt_plt-alt_rs))], '#06D6A0', label = 'SSP2-45')
    # plt.plot(year_rs.T, dens_ssp3_70_rs[:,np.argmin(np.abs(alt_plt-alt_rs))]/dens_rs[:,np.argmin(np.abs(alt_plt-alt_rs))], '#FFD166', label = 'SSP3-70')
    # plt.plot(year_rs.T, dens_ssp5_85_rs[:,np.argmin(np.abs(alt_plt-alt_rs))]/dens_rs[:,np.argmin(np.abs(alt_plt-alt_rs))], '#EF476F', label = 'SSP5-85')
    # plt.xlabel('Year')
    # plt.ylim([0,1])
    # # plt.ylabel('Density Multiplier')
    # plt.title('900 km')
    # plt.grid(axis = 'both', color = 'gainsboro', linewidth = 0.5)
    # plt.show()
    
    # make a subplot for each ssp case, showing the density multiplier as a function of altitude for each year    
    colors = ['#00798c', '#edae49', '#d1495b', 'k', '#6a4c93']
    plt.figure(figsize = (10,3))
    plt.subplot(1,3,1)
    plt.plot(year_rs.T, dens_ssp1_26_rs[:,np.argmin(np.abs(300-alt_rs))]/dens_rs[:,np.argmin(np.abs(300-alt_rs))], colors[0], label = '300')
    plt.plot(year_rs.T, dens_ssp1_26_rs[:,np.argmin(np.abs(1000-alt_rs))]/dens_rs[:,np.argmin(np.abs(1000-alt_rs))], colors[0], label = '1000', alpha = 0.4)
    plt.axvspan(2000, 2023.5, color = 'whitesmoke')
    plt.ylim([0,1])
    plt.xlim([2000, 2100])
    # remove tick marks from y axis
    plt.tick_params(axis='y', which='both', left=False, right=False)
    # remove tick marks from x axis
    # plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.xlabel('Year')
    # plt.title('SSP1-26')
    plt.legend(title = 'Altitude [km]', loc = 'lower left')
    plt.ylabel('Density Multiplier')
    plt.grid(axis = 'y', color = 'gainsboro', linewidth = 0.5)
    plt.xticks(rotation=45)

    
    plt.subplot(1,3,2)
    plt.plot(year_rs.T, dens_ssp2_45_rs[:,np.argmin(np.abs(300-alt_rs))]/dens_rs[:,np.argmin(np.abs(300-alt_rs))], colors[1], label = '300')
    plt.plot(year_rs.T, dens_ssp2_45_rs[:,np.argmin(np.abs(1000-alt_rs))]/dens_rs[:,np.argmin(np.abs(1000-alt_rs))], colors[1], label = '1000', alpha = 0.4)
    plt.axvspan(2000, 2023.5, color = 'whitesmoke')
    plt.ylim([0,1])
    plt.xlim([2000, 2100])
    # remove tick marks from y axis
    plt.tick_params(axis='y', which='both', left=False, right=False)
    # remove tick marks from x axis
    # plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.xlabel('Year')
    # plt.title('SSP2-45')
    plt.legend(title = 'Altitude [km]', loc = 'lower left')
    # plt.ylabel('Density Multiplier')
    plt.grid(axis = 'y', color = 'gainsboro', linewidth = 0.5)
    # get rid of x axis labels and share with the first plot
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    # set x axis labels to 45 degrees
    plt.xticks(rotation=45)
    
    plt.subplot(1,3,3)
    plt.plot(year_rs.T, dens_ssp5_85_rs[:,np.argmin(np.abs(300-alt_rs))]/dens_rs[:,np.argmin(np.abs(300-alt_rs))], colors[2], label = '300')
    plt.plot(year_rs.T, dens_ssp5_85_rs[:,np.argmin(np.abs(1000-alt_rs))]/dens_rs[:,np.argmin(np.abs(1000-alt_rs))], colors[2], label = '1000', alpha = 0.4)
    plt.axvspan(2000, 2023.5, color = 'whitesmoke')
    plt.ylim([0,1])
    plt.xlim([2000, 2100])
    # remove tick marks from y axis
    plt.tick_params(axis='y', which='both', left=False, right=False)
    # remove tick marks from x axis
    # plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.xlabel('Year')
    # plt.title('SSP1-26')
    plt.legend(title = 'Altitude [km]', loc = 'lower left')
    # plt.ylabel('Density Multiplier')
    plt.grid(axis = 'y', color = 'gainsboro', linewidth = 0.5)
    # get rid of x axis labels and share with the first plot
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # plot the density at 600 km for each SSP and the baseline
    plt.figure(figsize = (8,4))
    plt.semilogy(year_rs.T, dens_rs[:,np.argmin(np.abs(600-alt_rs))], color = 'k', label = 'Baseline', linestyle = ':')
    plt.semilogy(year_rs.T, dens_ssp1_26_rs[:,np.argmin(np.abs(600-alt_rs))],color = colors[0], label = 'SSP1-26')
    plt.semilogy(year_rs.T, dens_ssp2_45_rs[:,np.argmin(np.abs(600-alt_rs))], color = colors[1], label = 'SSP2-45')
    plt.semilogy(year_rs.T, dens_ssp5_85_rs[:,np.argmin(np.abs(600-alt_rs))], color = colors[2], label = 'SSP5-85')
    plt.semilogy(year_rs.T, dens_rs[:,np.argmin(np.abs(600-alt_rs))], color = 'k', linestyle = ':')
    plt.tick_params(axis='y', which='both', left=False, right=False)
    # shade everything from 1995 to 2023.5 in whitesmoke
    plt.axvspan(1995, 2023.5, color = 'whitesmoke')
    plt.xlim([1995,2105])
    plt.xlabel('Year')
    plt.ylabel('Atmospheric Mass Density [$\mathregular{kg/m^3}$]')
    plt.grid(axis = 'y', color = 'gainsboro')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_ssp_arrays(ssp_arrays, year_rs, alt_rs, ssp_array_names, levels=[1e-14, 1e-12, 1e-10], fmt='%e'):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 4, figsize=(15, 10))

    # Flatten the axs array for easier iteration
    axs = axs.ravel()

    # Define the normalization for the color scale
    vmin = min(ssp.min() for ssp in ssp_arrays)
    vmax = max(ssp.max() for ssp in ssp_arrays)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    i = 0
    for ax, ssp in zip(axs, ssp_arrays):
        # Plot the matrix with scaled x and y axis values
        cax = ax.imshow(ssp.T, extent=[year_rs.min(), year_rs.max(), alt_rs.min(), alt_rs.max()], 
                        origin='lower', aspect='auto', norm=norm, cmap='Greys')
        # Add contours
        contours = ax.contour(ssp.T, levels, extent=[year_rs.min(), year_rs.max(), alt_rs.min(), alt_rs.max()], 
                              origin='lower', colors='r', norm=norm, linewidths=(1,2,3))
        # Set the contour labels to use scientific notation
        fmt = ticker.LogFormatterMathtext()
        fmt.create_dummy_axis()
        ax.clabel(contours, inline=True, inline_spacing=20, fontsize=10, fmt=fmt)
        # Set the title
        ax.set_title(ssp_array_names[i])
        ax.set_xlabel('Year')
        ax.set_ylabel('Altitude [km]')
        i += 1

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.tight_layout()

    # Show the plot
    plt.show()

def compute_dens(f107, alt):
    # run NRLMSISE-00 for each year in f107_year_fit_proj with the alts from ALT -- sample across many lats and lons to get global average
    # get the multiplier for each year and altitude from NRLMSISE-00 compared to 2000, then compare to the RCP multipliers -- how off are they from where the natural deviation is?
    lats = np.arange(-90,90,30)
    lons = np.arange(0,360,30)
    ap = 0
    dens_p = np.zeros((len(lats), len(lons), len(alt), len(f107)))
    dens = np.zeros((len(f107),len(alt)))
    for i in range(len(f107)):
        for m in range(len(alt)):
            for j in range(len(lats)):
                for k in range(len(lons)):
                    # query nrlmsise-00 for this combination of lat, lon, and alt
                    nrl00 = msise_model(dt.datetime(2020,1,1), alt[m], lats[j], lons[k], f107[i], f107[i], ap)
                    dens_p[j,k,m,i] = nrl00[0][5]*1e2**3/1e3

            # take average dens_p for each altitude each year
            dens[i,m] = np.average(dens_p[:,:,m,i])
        print(i/len(f107))
    return dens
                
def project_f107():
    # load SW-All data from mat file
    column_to_read = 'F10.7_ADJ'
    df = pd.read_csv('../COLA/SW-All.csv', usecols=[column_to_read])
    f107_hist = np.array(df[column_to_read])[:24010]
    
    # f107_hist = data['sw_data'][1:-1400,26]
    date_hist = dt.datetime(1957,10,1) + np.array([dt.timedelta(days = int(i)) for i in range(len(f107_hist))])
    # create dt, days since 1957-10-01
    dt_hist = np.arange(0,len(f107_hist),1)

    # clean up f107 data -- for any value that is greater than 400, just set it to the value of the previous day
    for i in range(len(f107_hist)):
        if f107_hist[i] > 400:
            f107_hist[i] = f107_hist[i-1]
        
    # Define the function to fit F10.7 to a sinusoid
    def sinusoid(x, A, f, phi, c):
        return A * np.sin(2 * np.pi * f * x + phi) + c

    # get movmean of f107_hist
    df = pd.DataFrame({'f107_df': f107_hist})
    f107_hist_movmean = df['f107_df'].rolling(90).mean()
    f107_hist_movmean = list(f107_hist_movmean)
    
    # Fitting the data to the sinusoidal function
    x0 = np.array([0.8, 1/(11*365), 0, 2.15])
    params, covariance = curve_fit(sinusoid, dt_hist, np.log10(f107_hist), p0 = x0)

    # Getting the fitted curve
    fitted_curve = sinusoid(dt_hist, *params)
    # compute the 10th and 90th percentiles from the covariance and make a curve for each
    # get the 10th and 90th percentile of each parameter
    params_10 = params - 1.645*np.sqrt(np.diag(covariance))
    params_90 = params + 1.645*np.sqrt(np.diag(covariance))
    # make a curve for each
    fitted_curve_10 = sinusoid(dt_hist, *params_10)
    fitted_curve_90 = sinusoid(dt_hist, *params_90)
    
    # make days from 2000-01-01 to 2100-01-01
    # year_rs = np.arange(0, 101*365, 365.25)
    # date_hist_long = dt.datetime(2000,1,1) + np.array([dt.timedelta(days = int(year_rs[i])) for i in range(len(year_rs))])
    # take the last value of mult26, 45, 60, 85 and continue this value for all remaining time
    year_rs = np.arange((2000-1957.75085)*365.25,(2200-1957.75085)*365.25+1,30.436)
    date_hist_long = dt.datetime(1957,10,1) + np.array([dt.timedelta(days = int(year_rs[i])) for i in range(len(year_rs))])
    f107_rs = 10**(sinusoid(year_rs, *params))
    year_rs_all = np.arange(0,(2100-1957.75085)*365.25,30.436)
    date_rs_all = dt.datetime(1957,10,1) + np.array([dt.timedelta(days = int(year_rs_all[i])) for i in range(len(year_rs_all))])
    f107_rs_all = 10**(sinusoid(year_rs_all, *params))
    year_rs = year_rs - year_rs[0]
    year_rs_year = 2000 + year_rs/365.25
    
    # make f107_hist_combined be f107_hist_movmean until the last date where we have data, then switch to the fitted curve
    # get the start index of f107_hist_movmean that is after 2000-01-01
    start_index_hist = np.argmin([np.abs(((date_hist_item - dt.datetime(2000,1,1)).days)) for date_hist_item in date_hist])
    # get the end index of f107_hist_movmean that is before 2100-01-01
    end_index_hist = np.argmin([np.abs(((date_hist_item - dt.datetime(2100,1,1)).days)) for date_hist_item in date_hist])

    date_end_hist = date_hist[end_index_hist]
    # find the index of the day in date_hist_long that is the first after date_end_hist
    idx_start_fitted = np.argmin([np.abs((date_hist_long_item - date_end_hist).days) for date_hist_long_item in date_hist_long]) + 1 # +1 ensures that we're starting after the end of the measurments!
    idx_end_fitted = np.argmin([np.abs((date_hist_long_item - dt.datetime(2100,1,1)).days) for date_hist_long_item in date_hist_long])
    
    # make f107_hist_combined
    f107_hist_combined = np.concatenate((f107_hist_movmean[start_index_hist:end_index_hist], f107_rs[idx_start_fitted:idx_end_fitted]))
    date_hist_combined = np.concatenate((date_hist[start_index_hist:end_index_hist], date_hist_long[idx_start_fitted:idx_end_fitted]))
    # plt.figure()
    # plt.plot(f107_hist_combined)
    # plt.show()
    
    # plt.figure()
    # plt.plot(date_hist[start_index_hist:end_index_hist], f107_hist_movmean[start_index_hist:end_index_hist], 'r-')
    # plt.plot(date_hist_long[idx_start_fitted:idx_end_fitted], f107_rs[idx_start_fitted:idx_end_fitted], 'b-')
    # plt.ylabel('F10.7 [sfu]')
    # plt.xlabel('Year')
    # plt.grid(axis = 'both', color = 'lightgray')
    # plt.show()
        
    # plt.figure()
    # plt.plot(date_hist_combined, f107_hist_combined, '-')
    # plt.xlabel('Year')
    # plt.ylabel('F10.7 [sfu]')
    # plt.grid(axis = 'both', color = 'whitesmoke')
    # plt.show()
    
    plt.figure(figsize = (3,3))
    plt.plot(date_hist, f107_hist_movmean, color = 'darkgray', label = 'Observed', linewidth = 0.7)
    plt.plot(date_rs_all, f107_rs_all, 'r--', label = 'Fitted', linewidth = 0.7)
    plt.plot(date_hist_combined, f107_hist_combined, 'k-', label = 'Modeled', linewidth = 0.7)
    # fill a gray background until 2023-07
    plt.axvspan(dt.datetime(1955,1,1), dt.datetime(2023,7,1), color = 'whitesmoke')
    plt.xlabel('Year')
    plt.ylabel('F$_{10.7}$ [sfu]')
    plt.xlim([dt.datetime(1955,1,1), dt.datetime(2105,1,1)])
    plt.grid(axis = 'both', color = 'gainsboro', linewidth = 0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return date_hist_combined, f107_hist_combined

if __name__ == '__main__':
    main()
