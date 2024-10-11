'''
FILE DESCRIPTION: This script computes the optimal distribution of characteristic satellites across altitude shells
of interest that maximizes the number of satellites while preventing long-term debris instability. Equilibrium points 
are found for each shell according to a 2-species source-sink model. We start at the top shell, then use the number of 
characteristic satellites in that shell to compute a debris flux rate into the lower shells. Since the optimal strategy
for maximizing the number of satellites involves balancing adding new satellites and minimizing debris flux, an iterative
optimization scheme is used to assign the best weights on the populations within each shell. 

This script is computationally intensive and may take some time to run on systems with limited compute resources. Modifying 
the time variable 'year' to cover less time or make larger timesteps will help to get output in reasonable time. Parallel 
computing toolboxes are used to speed up the computation as best as possible. 

05/01/2024 - William Parker
'''

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize, fsolve, root_scalar
import copy
from scipy.io import loadmat, savemat
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
import time 
import pickle as pkl

# set font to arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
# make font gray
plt.rcParams['text.color'] = 'gray'
plt.rcParams['axes.labelcolor'] = 'gray'
plt.rcParams['xtick.color'] = 'gray'
plt.rcParams['ytick.color'] = 'gray'
# make the box around the plot gray
plt.rcParams['axes.edgecolor'] = 'white'

def main():
    # set initial parameters
    r_e = 6378.137 # Earth radius in km
    mu = 398600 # Earth gravitational parameter in km^3/s^2
    r = np.array([0.000745, 0.00009]) # km, for sat, deb respectively
    m = np.array([223,0.64])
    cd = 2.2
    B = np.pi*r**2*cd/m # ballistic coefficient in km^2/kg
    B_deb = B[0]
    
    # Create a design variable s0_dist that is a distribution of initial altitudes for satellites within the provided bins
    bins = np.arange(200,1000+1,20)

    r_sat = r[0] # km
    r_deb = r[1] # km
    sigma = np.array([[(r_sat + r_sat)**2, (r_sat + r_deb)**2], [(r_sat + r_deb)**2, (r_deb + r_deb)**2]])
    
    # compute the volume of the space in each bin 
    vol_bin = [4/3*np.pi*((bins[i+1]+r_e)**3 - (bins[i]+r_e)**3) for i in range(len(bins)-1)] 
    # compute average velocity from center altitude value in each bin
    v_bin = np.sqrt(mu/(r_e + (bins[1:] + bins[:-1])/2)) 

    # compute the 2d array for phi for each bin
    phi_bin = np.zeros((len(bins)-1, 2,2))
    for i in range(len(bins)-1):
        phi_bin[i,:,:] = np.pi*sigma*v_bin[i]/vol_bin[i]

    n_frag = np.array([[497.7384,116.0473],[116.0473,116.0473]]) # values taken from MOCAT SSEM using NASA collision breakup model and sat/deb radii and masses (masses presumed from BC). 


    # load processed file with time and altitude-resolved density profiles for each of the SSPs. 
    ip_file = 'data/dens_forecast_ssp_v2_msis2.pkl'
    op_file = 'data/opt_mult_contraction_ssp_hd_v2.mat'

    # load data from pkl
    # with open(processed_file, 'wb') as f:
    # pkl.dump([dens_ssp1_19_rs, dens_ssp1_26_rs, dens_ssp2_45_rs, dens_ssp3_70_rs, dens_ssp3_70_lowNTCF_rs, dens_ssp4_34_rs, dens_ssp4_60_rs, dens_ssp5_34_over_rs, dens_ssp5_85_rs, dens_rs, alt_rs, year_rs], f)
    data = pkl.load(open(ip_file, 'rb'))
    dens_ssp1_19_rs = data[0]
    dens_ssp1_26_rs = data[1]
    dens_ssp2_45_rs = data[2]
    dens_ssp3_70_rs = data[3]
    dens_ssp3_70_lowNTCF_rs = data[4]
    dens_ssp4_34_rs = data[5]
    dens_ssp4_60_rs = data[6]
    dens_ssp5_34_over_rs = data[7]
    dens_ssp5_85_rs = data[8]
    dens_rs = data[9]
    alt_rs = data[10]
    year_rs = data[11]
    
    # load dens_contraction
    # ssp_dens = loadmat('data/dens_forecast_ssp.mat')
    # dens_ssp1_26 = ssp_dens['dens_ssp1_26_rs']
    # dens_ssp2_45 = ssp_dens['dens_ssp2_45_rs']
    # dens_ssp5_85 = ssp_dens['dens_ssp5_85_rs']
    # dens_baseline = ssp_dens['dens_rs']
    # alt_rs = ssp_dens['alt_rs']
    # year_rs = ssp_dens['year_rs']
    
    dens_profile = [ 'dynamic_baseline', 'ssp1_26', 'ssp2_45', 'ssp5_85']
    dens_contraction = {}
    dens_contraction['dynamic_baseline'] = dens_rs
    dens_contraction['ssp1_26'] = dens_ssp1_26_rs
    dens_contraction['ssp2_45'] = dens_ssp2_45_rs
    dens_contraction['ssp5_85'] = dens_ssp5_85_rs
    dens_contraction['alt_rs'] = alt_rs
    dens_contraction['year_rs'] = year_rs
    
    # set timestep in years. This is a fine timestep and will take some time to evolve.
    year = np.arange(2000,2100 + 1/4, 1/4)

    # check to see if opt_mult.mat exists, and if so, load variables (make sure this is the opt_mult file you want!)
    try:
        data = loadmat(op_file)
        s_per_bin = data['s_per_bin']
        n_per_bin = data['n_per_bin']
        
    except:
        
        # start at the top shell, find the optimal mult value for this shell when all shells below are filled to capacity. (Max total sats)
        # Once we run through this once, we re-run again with the values below until we converge on multiplier values that are optimal.
        n_cycle = 5
        
        mult_vec = np.zeros(( len(year), len(bins)-1, n_cycle, len(dens_profile)))
        s_per_bin = np.zeros((len(year), len(dens_profile),  len(bins)-1))
        n_per_bin = np.zeros((len(year), len(dens_profile),  len(bins)-1))
        
        # check1, check2, check3 = prop_cap_over_profiles(1, year, dens_profile, n_cycle, bins, B_deb, mu, r_e, dens_contraction, phi_bin, n_frag)
        t0 = time.time()
        results = Parallel(n_jobs=cpu_count())(delayed(prop_cap_over_profiles)(n, year, dens_profile, n_cycle, bins, B_deb, mu, r_e, dens_contraction, phi_bin, n_frag) for n in range(len(year)))
        t1 = time.time()
        print(t1-t0)
        for i in range(len(results)):
            mult_vec[i,:,:,:], s_per_bin[i,:,:], n_per_bin[i,:,:] = results[i]
        
        # save the optimal mult values and the s_per_bin and n_per_bin values
        savemat(op_file, {'opt_mult': mult_vec, 's_per_bin': s_per_bin, 'n_per_bin': n_per_bin, 'bins': bins, 'mult_vec': mult_vec})
    
    s_sum_per_year = np.zeros((len(year), len(dens_profile)))
    n_sum_per_year = np.zeros((len(year), len(dens_profile)))
    for i in range(len(year)):
        for j in range(len(dens_profile)):
            s_sum_per_year[i,j] = np.sum(s_per_bin[i,j,:])
            n_sum_per_year[i,j] = np.sum(n_per_bin[i,j,:])
            
    # plot the s_sum_per_year values for each dens_profile over each year
    plt.figure(figsize = (6,5))
    start_year = 1947.7
    end_year = 2105
    interval = 10.93
    current_year = start_year
    while current_year <= end_year:
        # given the year in decimal format, create the corresponding datetime object with the months and days necessary to account for the decimal
        # year = int(current_year)
        # month = int((current_year - year) * 12) + 1
        # day = 1
        plt.axvline(current_year, color='gainsboro', linestyle='-', linewidth=0.5)
        current_year += interval
    colors =  ['k', '#00798c', '#edae49', '#d1495b']
    dens_profile_plot = [ 'Baseline', 'SSP1-26', 'SSP2-45', 'SSP5-85']
    for i in range(len(dens_profile)):
        # plt.figure(figsize = (6,5))
        if i == 0:
            plt.plot(year, s_sum_per_year[:,i], 'k:', label = dens_profile_plot[i], linewidth = 1)
        else: 
            plt.plot(year, s_sum_per_year[:,i], '-', label = dens_profile_plot[i], linewidth = 1.5, color = colors[i])
    
    plt.plot(year, s_sum_per_year[:,0], 'k:', linewidth = 1)
    plt.axvspan(2000, 2023.5, color = 'whitesmoke')
    plt.legend()
    plt.grid(axis = 'y', color = 'lightgray')
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel('Year')
    plt.ylabel('Number of Satellites')
    plt.xlim([2000,2100])
    plt.tight_layout()
    plt.show()
            
    plt.figure()
    plt.grid(axis = 'both', color = 'lightgray', linewidth = 0.25)
    plt.hist(bins[:-1], bins=bins, weights=n_per_bin[0,0,:], label='Debris', alpha=0.5, color = 'tab:orange')
    plt.hist(bins[:-1], bins=bins, weights=s_per_bin[0,0,:], label='Satellites', alpha=0.5, color = 'tab:blue')
    plt.yscale('log')
    plt.xlabel('Shell Lower Altitude [km]')
    plt.legend()
    plt.ylabel('Number of Satellites')    
    plt.show()

    
def prop_cap_over_profiles(n, year, dens_profile, n_cycle, bins, B_deb, mu, r_e, dens_contraction, phi_bin, n_frag): 
    mult_vec_c = np.zeros((len(bins)-1, n_cycle,len(dens_profile)))
    s_per_bin_c = np.zeros((len(dens_profile), len(bins)-1))
    n_per_bin_c = np.zeros((len(dens_profile), len(bins)-1))
    mult = np.zeros((len(bins)-1))
    for m in range(len(dens_profile)):
        # propagate a debris object forward in time from the top of each bin until the bottom and look at the time profile (use this to determine the acceptable amount of debris)
        t_deb_decay = deb_decay_t_per_bin(bins, B_deb, mu, r_e, dens_contraction, dens_profile[m], year[n])  
        for j in range(n_cycle):
            for k in range(len(bins)-1):
                mult_opt = minimize(compute_neg_s0_from_mult_scalar, args = (mult, n_frag, phi_bin, t_deb_decay, bins, k), method ='Nelder-Mead', tol = 1e-5, x0 = 0.5, bounds = ((0,1),))#, options={'disp': True} )
                mult[k] = mult_opt.x
                
            mult_vec_c[:,j,m] = copy.deepcopy(mult)
            
        # use the optimal mult values to compute overall capacity by shell -- take the last iteration's mult values as the best
        s_per_bin_c[m,:], n_per_bin_c[m,:] = get_s0_and_n0_from_mult(mult_vec_c[:,-1,m], n_frag, phi_bin, t_deb_decay, bins)
        
        # reset mult to zero to start
        mult = np.zeros((len(bins)-1))
        
        print(m/len(dens_profile))
        
    print(n/len(year))
    return mult_vec_c, s_per_bin_c, n_per_bin_c
    
def get_s0_and_n0_from_mult(mult, n_frag, phi_bin, t_deb_decay, bins, plot_figs = False):
    N_dot_sum_above = 0
    s0_eq = np.zeros((len(bins)-1,))
    n0_eq = np.zeros((len(bins)-1,))
    n_dot_eq = np.zeros((len(bins)-1,))
    deb_leave_rate = np.zeros((len(bins),))
    N_dot_sum_above_check = np.zeros((len(bins)-1,))
    # find the value for the array mult that maximizes the number of satellites that can be fit across all shells
    for k in range(len(bins)-1):
        k = len(bins)-1-k-1 # go in reverse order
        #n1 == n2 when b**2 - 4*a*c = 0 -- find the value of s that minimizes b**2 - 4*a*c
        s_min = fsolve(quad_solve, args = (n_frag, phi_bin, t_deb_decay, N_dot_sum_above, k), x0 = 1)
        s0_eq[k] = s_min[0]*mult[k]
        n0_eq[k], n_dot_eq[k], deb_leave_rate[k] = quad_solve_n(s0_eq[k], n_frag, phi_bin, t_deb_decay, N_dot_sum_above, k)
        
        N_dot_sum_above = deb_leave_rate[k]
        N_dot_sum_above_check[k] = N_dot_sum_above
    
    if plot_figs == True: 
        plt.figure()
        plt.semilogy(bins[:-1], s0_eq, '.')
        plt.semilogy(bins[:-1], n0_eq, '.')
        plt.grid(axis = 'both', color = 'lightgray')
        plt.xlabel('Shell Lower Altitude [km]')
        plt.ylabel('Number of Satellites')
        
        plt.figure()
        plt.semilogy(bins[:-1], N_dot_sum_above_check, '.')
        plt.xlabel('Shell Lower Altitude [km]')
        plt.ylabel('N_dot_sum_above')
        
        plt.figure()
        plt.semilogy(bins[:-1], n_dot_eq, 'r.')
        plt.ylabel('n_dot_eq')
        plt.show()
        
    return s0_eq, n0_eq


def compute_neg_s0_from_mult_scalar(mult_scalar, mult_vec, n_frag, phi_bin, t_deb_decay, bins, bin_idx, plot_figs = False):
    mult = copy.deepcopy(mult_vec)
    mult[bin_idx] = mult_scalar
    
    N_dot_sum_above = 0
    s0_eq = np.zeros((len(bins)-1,))
    n0_eq = np.zeros((len(bins)-1,))
    n_dot_eq = np.zeros((len(bins)-1,))
    deb_leave_rate = np.zeros((len(bins),))
    N_dot_sum_above_check = np.zeros((len(bins)-1,))
    # find the value for the array mult that maximizes the number of satellites that can be fit across all shells
    for k in range(len(bins)-1):
        k = len(bins)-1-k-1 # go in reverse order
        #n1 == n2 when b**2 - 4*a*c = 0 -- find the value of s that minimizes b**2 - 4*a*c
        s_min = fsolve(quad_solve, args = (n_frag, phi_bin, t_deb_decay, N_dot_sum_above, k), x0 = 1)
        # plot_n_dot_vs_s(n_frag, phi_bin, t_deb_decay, k)
        s0_eq[k] = np.max([s_min[0]])*mult[k]
        n0_eq[k], n_dot_eq[k], deb_leave_rate[k] = quad_solve_n(s0_eq[k], n_frag, phi_bin, t_deb_decay, N_dot_sum_above, k)
        
        N_dot_sum_above = deb_leave_rate[k]
        N_dot_sum_above_check[k] = N_dot_sum_above

    
    if plot_figs == True: 
        plt.figure()
        plt.semilogy(bins[:-1], s0_eq, '.')
        plt.semilogy(bins[:-1], n0_eq, '.')
        plt.grid(axis = 'both', color = 'lightgray')
        plt.xlabel('Shell Lower Altitude [km]')
        plt.ylabel('Number of Satellites')
        
        plt.figure()
        plt.hist(s0_eq, bins = bins, label = 'Satellites')
        plt.hist(n0_eq, bins = bins, label = 'Debris')
        plt.xlabel('Shell Lower Altitude [km]')
        plt.ylabel('Number of Satellites')
        
        plt.figure()
        plt.semilogy(bins[:-1], N_dot_sum_above_check, '.')
        plt.xlabel('Shell Lower Altitude [km]')
        plt.ylabel('N_dot_sum_above')
        
        plt.figure()
        plt.semilogy(bins[:-1], n_dot_eq, 'r.')
        plt.ylabel('n_dot_eq')
        plt.show()
        
    # print(np.sum(s0_eq))
    return(-np.sum(s0_eq))
    
def quad_solve(s0, n_frag, phi_bin, t_deb_decay, N_dot_sum_above, k):
    a = phi_bin[k,1,1]*n_frag[1,1]
    b = phi_bin[k,0,1]*n_frag[0,1]*s0 - 1/t_deb_decay[k]
    c = phi_bin[k,0,0]*n_frag[0,0]*s0**2 + N_dot_sum_above
    #n1 == n2 when b**2 - 4*a*c = 0 -- find the value of s that minimizes b**2 - 4*a*c
    resid = np.abs(b**2 - 4*a*c)
    # val_eq = b**2/(2*a)
    return resid  

def quad_solve_n(s0, n_frag, phi_bin, t_deb_decay, N_dot_sum_above, k):
    a = phi_bin[k,1,1]*n_frag[1,1]
    b = phi_bin[k,0,1]*n_frag[0,1]*s0 - 1/t_deb_decay[k]
    c = phi_bin[k,0,0]*n_frag[0,0]*s0**2 + N_dot_sum_above
    # n = (-b)/(2*a) # b**2 - 4*a*c == 0
    if b**2 - 4*a*c > 0:
        n = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    else:
        n = (-b)/(2*a) # if b**2 - 4*a*c is negative, then we're in the regime where n1 and n2 are complex, so we just take the real part of n1
    n_dot = phi_bin[k,1,1]*n_frag[1,1]*n**2 + phi_bin[k,0,1]*n_frag[0,1]*s0*n + phi_bin[k,0,0]*n_frag[0,0]*s0**2 - 1/t_deb_decay[k]*n + N_dot_sum_above
    deb_remove_rate = 1/t_deb_decay[k]*n
    return n, n_dot, deb_remove_rate

def deb_decay_t_per_bin(bins, B, mu, r_e, dens_contraction, dens_profile, year, dens_mult = 1):
    # key assumption here is that decay rate is constant throughout the bin. This is valid if the bins are small but you should consider this carefully and test to make sure it is a reasonable assumption. 
    t_deb_decay = np.zeros((len(bins)-1),)
    bin_width = bins[1]-bins[0]
    alts = bins[0:-1] + 0.5*bin_width
    v = np.sqrt(mu/(r_e+alts))
    decay_time = np.zeros(len(alts))
    for i in range(len(alts)):
        a_drag = -0.5  * den(alts[i], dens_contraction, dens_profile, year, dens_mult)*1e9 * v[i]**2 *B
        T = 2*np.pi*np.sqrt((r_e+alts[i])**3/mu)
        dalt_dt = a_drag*T/np.pi #https://en.wikipedia.org/wiki/Orbital_decay
        decay_time[i] = (bin_width/2)/np.abs(dalt_dt)
    
    return(decay_time)

def den(h_ellp, dens_contraction, dens_profile, year, dens_mult = 1):
    # commpute the distribution of these objects as a function of time, assuming a static exponential atmosphere
    year_idx = np.argmin(np.abs(dens_contraction['year_rs'] - year))
    alt_idx = np.argmin(np.abs(dens_contraction['alt_rs'] - h_ellp))
    if dens_profile == 'dynamic_baseline':
        dens_mtx = dens_contraction['dynamic_baseline']
        dens = dens_mtx[year_idx, alt_idx]*dens_mult
        return dens
    if dens_profile == 'ssp1_26':
        dens_mtx = dens_contraction['ssp1_26']
        dens = dens_mtx[year_idx, alt_idx]*dens_mult
        return dens
    elif dens_profile == 'ssp2_45':
        dens_mtx = dens_contraction['ssp2_45']
        dens = dens_mtx[year_idx, alt_idx]*dens_mult
        return dens
    elif dens_profile == 'ssp5_85':
        dens_mtx = dens_contraction['ssp5_85']
        dens = dens_mtx[year_idx, alt_idx]*dens_mult
        return dens
    elif dens_profile == 'ssp1_26_low':
        dens_mtx = dens_contraction['ssp1_26_low']
        dens = dens_mtx[year_idx, alt_idx]*dens_mult
        return dens
    elif dens_profile == 'ssp1_26_high':
        dens_mtx = dens_contraction['ssp1_26_high']
        dens = dens_mtx[year_idx, alt_idx]*dens_mult
        return dens
    elif dens_profile == 'ssp2_45_low':
        dens_mtx = dens_contraction['ssp2_45_low']
        dens = dens_mtx[year_idx, alt_idx]*dens_mult
        return dens
    elif dens_profile == 'ssp2_45_high':
        dens_mtx = dens_contraction['ssp2_45_high']
        dens = dens_mtx[year_idx, alt_idx]*dens_mult
        return dens
    elif dens_profile == 'ssp5_85_low':
        dens_mtx = dens_contraction['ssp5_85_low']
        dens = dens_mtx[year_idx, alt_idx]*dens_mult
        return dens
    elif dens_profile == 'ssp5_85_high':
        dens_mtx = dens_contraction['ssp5_85_high']
        dens = dens_mtx[year_idx, alt_idx]*dens_mult
        return dens
    

    elif dens_contraction == 'static':
        #~ Values taken from Vallado's "Fundamentals of Astrodynamics 
        #~ and Applications (2nd Edition)". Page 537, table 8-4
        #~ 
        #~ Remigiusz Pospieszynski
        #~ MMXVI
        if h_ellp >= 0  and h_ellp < 25 :
            h_0 = 0
            rho_0 = 1.225
            H = 7.249		
        if h_ellp >= 25 and h_ellp < 30 :
            h_0 = 25
            rho_0 = 3.899*10**(-2)
            H = 6.349		
        if h_ellp >= 30 and h_ellp < 40 :
            h_0 = 30
            rho_0 = 1.774*10**(-2)
            H = 6.682
        if h_ellp >= 40 and h_ellp < 50 :
            h_0 = 40
            rho_0 = 3.972*10**(-3)
            H = 7.554
        if h_ellp >= 50 and h_ellp < 60 :
            h_0 = 50
            rho_0 = 1.057*10**(-3)
            H = 8.382
        if h_ellp >= 60 and h_ellp < 70 :
            h_0 = 60
            rho_0 = 3.206*10**(-4)
            H = 7.714
        if h_ellp >= 70 and h_ellp < 80 :
            h_0 = 70
            rho_0 = 8.77*10**(-5)
            H = 6.549
        if h_ellp >= 80 and h_ellp < 90 :
            h_0 = 80
            rho_0 = 1.905*10**(-5)
            H = 5.799
        if h_ellp >= 90 and h_ellp < 100 :
            h_0 = 90
            rho_0 = 3.396*10**(-6)
            H = 5.382
        if h_ellp >= 100 and h_ellp < 110 :
            h_0 = 100
            rho_0 = 5.297*10**(-7)
            H = 5.877
        if h_ellp >= 110 and h_ellp < 120 :
            h_0 = 110
            rho_0 = 9.661*10**(-8)
            H = 7.263
        if h_ellp >= 120 and h_ellp < 130 :
            h_0 = 120
            rho_0 = 2.438*10**(-8)
            H = 9.473												
        if h_ellp >= 130 and h_ellp < 140 :
            h_0 = 130
            rho_0 = 8.484*10**(-9)
            H = 12.636
        if h_ellp >= 140 and h_ellp < 150 :
            h_0 = 140
            rho_0 = 3.845*10**(-9)
            H = 16.149
        if h_ellp >= 150 and h_ellp < 180 :
            h_0 = 150
            rho_0 = 2.070*10**(-9)
            H = 22.523
        if h_ellp >= 180 and h_ellp < 200 :
            h_0 = 180
            rho_0 = 5.464*10**(-10)
            H = 29.74
        if h_ellp >= 200 and h_ellp < 250 :
            h_0 = 200
            rho_0 = 2.789*10**(-10)
            H = 37.105
        if h_ellp >= 250 and h_ellp < 300 :
            h_0 = 250
            rho_0 =7.248*10**(-11)
            H = 45.546
        if h_ellp >= 300 and h_ellp < 350 :
            h_0 = 300
            rho_0 = 2.418*10**(-11)
            H = 53.628
        if h_ellp >= 350 and h_ellp < 400 :
            h_0 = 350
            rho_0 = 9.518*10**(-12)
            H = 53.298
        if h_ellp >= 400 and h_ellp < 450 :
            h_0 = 400
            rho_0 = 3.725*10**(-12)
            H = 58.515
        if h_ellp >= 450 and h_ellp < 500 :
            h_0 = 450
            rho_0 = 1.585*10**(-12)
            H = 60.828
        if h_ellp >= 500 and h_ellp < 600 :
            h_0 = 500
            rho_0 = 6.967*10**(-13)
            H = 63.822
        if h_ellp >= 600 and h_ellp < 700 :
            h_0 = 600
            rho_0 = 1.454*10**(-13)
            H = 71.835
        if h_ellp >= 700 and h_ellp < 800:
            h_0 = 700
            rho_0 = 3.614*10**(-14)
            H = 88.667
        if h_ellp >= 800 and h_ellp < 900 :
            h_0 = 800
            rho_0 = 1.17*10**(-14)
            H = 124.64
        if h_ellp >= 900 and h_ellp < 1000 :
            h_0 = 900
            rho_0 = 5.245*10**(-15)
            H = 181.05
        if h_ellp >= 1000:
            h_0 = 1000
            rho_0 = 3.019*10**(-15)
            H = 268									
        return rho_0*math.exp(-(h_ellp-h_0)/H)*dens_mult
    else: 
        print('dens_profile must be either dynamic_baseline, rcp26, rcp45, rcp60, rcp85, or static')

main()