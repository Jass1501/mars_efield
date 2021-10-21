#======================= Import python packages ================================
import numpy as np                          # for array operations
import matplotlib.pyplot as plt             # python plotting library
import csv                                  # handling csv data files
import os                                   # os command handling
from tqdm import tqdm                       # progress bar
from netCDF4 import Dataset                 # to read .nc files
from scipy.integrate import solve_ivp       # integration function
from functions import *                     # utility functions
#===============================================================================

#========================= Lectura de datos ====================================
f_zagl = Dataset('./00000.atmos_daily_zagl.nc', 'r') # Lectura de datos

#               ========== zagl: 34, lon: 60, lat: 35 ==========
lon      = f_zagl.variables['lon'][:]      # ('lon',), longitude [degrees east]
lat      = f_zagl.variables['lat'][:]      # ('lat',), latitude  [degrees north]
time     = f_zagl.variables['time'][:]     # ('time',), sol number  [days since 0000-00-00 00:00:00]

zagl     = f_zagl.variables['zagl'][:]     # ('zagl',), altitude above ground level [m]

dst_mass = f_zagl.variables['dst_mass'][:] # ('time', 'zagl', 'lat', 'lon'), dust aerosol mass mixing ratio [kg/kg]
dst_num  = f_zagl.variables['dst_num'][:]  # ('time', 'zagl', 'lat', 'lon'), dust aerosol number [number/kg]
rho      = f_zagl.variables['rho'][:]      # ('time', 'zagl', 'lat', 'lon'), density [kg/m3]

ucomp = f_zagl.variables['ucomp'][:]       # ('time', 'zagl', 'lat', 'lon'), zonal wind [m/s]
vcomp = f_zagl.variables['vcomp'][:]       # ('time', 'zagl', 'lat', 'lon'), meridional wind [m/s]

f_zagl.close()
#===============================================================================

#========================= Indices =============================================
zagl_index = 1
#===============================================================================

#========================= Volumen =============================================
x = longitude_halfcircle() # [m]
y = latitude_fullcircle()  # [m]
volume = compute_volume(y, x, zagl[1])
#===============================================================================

#========================= Radios de granos ====================================
n_points = 5000
points_r = np.linspace(1e-6, 50e-6, n_points) # 1 um <--> 50 um

    #===== Radio efectivo y varianza efectiva [madeleine,2011] ==
rad_eff = 1e-6
var_eff = 0.3

    #===== Radio medio y desviacion estandar ====================
sigma0 =  0.63676 # compute_sigma0(var_eff)

    #===== Matrices vacias para guardar datos ====================
r_dist   = np.zeros((35, 60, n_points)) # Matriz de dist de radios para cada bloque
radii_sl = np.zeros((35, 60, 2))        # Matriz de radios pequeños y grandes para cada bloque
mass_sl  = np.zeros((35, 60, 2))        # Matriz de masas pequeñas y grandes para cada bloque
dq_sl    = np.zeros((35, 60, 2))        # Matriz de cargas igual y diferente composicion para cada bloque
n_m3_sl  = np.zeros((35, 60, 2))        # Matriz de densidad de particulas por m3 para cada bloque
u_comp   = np.zeros((35, 60))        # Matriz de velocidades zonales
v_comp   = np.zeros((35, 60))        # Matriz de velocidades meridionales

# len(latitud) y len(longitud)
lat_len  = len(lat)
lon_len  = len(lon)
zagl_len = len(zagl[1:16])
time_len = len(time[:16])


dh = np.diff(time)[0] # delta hours
n_time = 60

#Parametros
eps_knot = 8.85e-12 # F/m
atm_cond_m = 2.5e-12 # S/m
phi = 2 # [V]

#Condiciones iniciales
y_dqeq = [0, 0]
y_dqnq = [0, 0]

#Data
efield_t_dqeq = []
efield_y_dqeq = []

efield_t_dqnq = []
efield_y_dqnq = []

#File counting variable
file_count = 1

#matrix to save names
name_matrix = np.linspace(1, 2100, 2100).reshape(35, 60)

# for lat_j in tqdm(range(lat_len)):
for lat_j in range(lat_len):
    for lon_i in range(lon_len):
        #===== printing estado ======================================
        print(f'Lat: {lat[lat_j]} and Lon: {lon[lon_i]}')
        w1 = 0 # initializing wind_speed
        
        for zi in range(1, zagl_len+1):
            
            for ti in range(time_len):

                # print(lat_j, lon_i, zi, ti)
            
                #===== variabes de bucle ====================================
                ti    = ti
                lat_j = lat_j # 29
                lon_i = lon_i # 0
                zi    = zi
                
                #===== masa del bloque ======================================
                block_mass = rho[ti, zi] * volume # [kg]

                #===== masa total de granos de polvo ========================
                m_dust = dst_mass[ti, zi] * block_mass # [kg]

                #===== numero de particulas de polvo ========================
                number_particles = dst_num[ti, zi] * block_mass # [number]
                # print(number_particles)
                
                #===== r0 ===================================================
                r0 = compute_rmean(m_dust, number_particles, sigma0)
                
                r00 = r0[lat_j, lon_i]
                
                if r00 >= points_r[-1] or r00 <= points_r[0]:
                    r00 = np.median(points_r)
                
                if abs(1e-6 - r00) < 10e-6:
                    r00 = 10e-6
                elif abs(50e-6 - r00) < 10e-6:
                    r00 = 10e-6

                #===== distribucion de granos log-normal() normalizada ======
                particle_distribution = grain_log_normal(points_r, r00, sigma0)
                particle_distribution = particle_distribution/particle_distribution.sum()

                #===== r_small y r_large ====================================
                r_dist_small, r_dist_large, l_points, r_points = dist_division(points_r, r00, sigma0)

                #===== n(r), distribucion de particulas =====================
                r_dist[lat_j, lon_i] = number_particles[lat_j, lon_i] * particle_distribution
                
                r_dist_index = r_dist[lat_j, lon_i]

                le_weight = r_dist_index[l_points]
                ri_weight = r_dist_index[r_points]
                
                #===== radio pequeño y radio grande =========================
                r_small = np.average(r_dist_small, weights=le_weight) # [m]
                r_large = np.average(r_dist_large, weights=ri_weight) # [m]
                
                radii_sl[lat_j, lon_i] = r_small, r_large

                #===== masa individual de los granos ========================
                mass_small = grain_mass(r_small) # [kg]
                mass_large = grain_mass(r_large) # [kg]

                # mass_sl[lat_j, lon_i] = mass_small, mass_large

                #===== intercambio de Carga =================================
                dqeq = dq_eq(r_small)
                dqnq = dq_nq(phi, r_small, r_large)

                dq_sl[lat_j, lon_i] = dqeq, dqnq

                #===== densidad de granos ===================================
                N_particles = np.sum(r_dist_index[l_points]) + np.sum(r_dist_index[r_points])

                small_particles_per_m3 = particles_density(N_particles, volume[lat_j, lon_i]) # [1/m3]
                large_particles_per_m3 = particles_density(N_particles, volume[lat_j, lon_i]) # [1/m3]

                n_m3_sl[lat_j, lon_i] = small_particles_per_m3, large_particles_per_m3
                
                #===== velocidad del viento =================================
                dudx = d_dvar(ucomp[ti, zi], (lat_j, lon_i), 'dx', y[lat_j, lon_i])
                dvdy = d_dvar(vcomp[ti, zi], (lat_j, lon_i), 'dy', x)

                h_div = dudx + dvdy

                dz_zagl = zagl[zi] - zagl[zi-1] # diferencia de altura
                w_speed = w1 - dz_zagl*h_div

                # print(w_speed)

                w1 = w_speed

                #===== velocidad de granos ==================================
                v_small_grains = particles_speed(rho[ti, zi, lat_j, lon_i], w_speed, r_small, mass_small)  # [m/s]
                v_large_grains = particles_speed(rho[ti, zi, lat_j, lon_i], w_speed, r_large, mass_large)  # [m/s]
                
                #===== delta velocidad ======================================
                delta_v = v_large_grains - v_small_grains # [m/s] # vlarge - vsmall
                
                #===== frecuencia de colision ===============================
                freq_col = np.pi * r_large**2 * delta_v * small_particles_per_m3

                #===== integracion campo electrico ==========================
                #===== matrices =============================================
                a_matrix = np.array([[0, 1], [0, -atm_cond_m/eps_knot]])

                k_eq = (small_particles_per_m3/eps_knot) * delta_v * freq_col * dqeq
                k_nq = (small_particles_per_m3/eps_knot) * delta_v * freq_col * dqnq

                #===== solucion ecuacion diferencial ========================
                dt = np.linspace(time[ti], dh + time[ti], n_time) # Dias

                mars_dqeq = solve_ivp(lambda t, y: g(t, y, a_matrix, k_eq), [dt[0], dt[-1]], y_dqeq, t_eval=dt, rtol=1e-5)
                mars_dqnq = solve_ivp(lambda t, y: g(t, y, a_matrix, k_nq), [dt[0], dt[-1]], y_dqnq, t_eval=dt, rtol=1e-5)
                
                y_dqeq = mars_dqeq.y[0][-1], mars_dqeq.y[1][-1]
                y_dqnq = mars_dqnq.y[0][-1], mars_dqnq.y[1][-1]

                efield_t_dqeq.append(mars_dqeq.t)
                efield_y_dqeq.append(mars_dqeq.y[0])

                efield_t_dqnq.append(mars_dqnq.t)
                efield_y_dqnq.append(mars_dqnq.y[0])

            #===== guardar datos =======================================
            efield_t_dqeq = np.concatenate(efield_t_dqeq)
            efield_y_dqeq = np.concatenate(efield_y_dqeq)
            
            efield_t_dqnq = np.concatenate(efield_t_dqnq)
            efield_y_dqnq = np.concatenate(efield_y_dqnq)

            #===== guardar dato ========================================
            folder_save = './data_efield_5/'
            counter = int(name_matrix[lat_j, lon_i])
            ff_save = f'folder{counter:04d} {int(lat[lat_j])}, {int(lon[lon_i])}/'
            
            try:
                os.mkdir(folder_save + ff_save)
            except:
                pass

            with open(folder_save + ff_save + f'zagl{zagl[zi]}.csv', 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows(zip(efield_t_dqeq, efield_y_dqeq, efield_t_dqnq, efield_y_dqnq))

            #===== reset condiciones iniciales =========================
            efield_t_dqeq = []
            efield_y_dqeq = []
            y_dqeq = [0, 0]

            efield_t_dqnq = []
            efield_y_dqnq = []
            y_dqnq = [0, 0]

        # file_count += 1
        # exit()
#===============================================================================
print('Ejecucion completada satisfactoriamente')
#========================= Finalizacion campo electrico ========================