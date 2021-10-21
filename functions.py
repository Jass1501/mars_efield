import numpy as np

#============= Constantes =================================
rhodust = 2.5e3 # [kg/m3], densidad de los granos de polvo
grav = 3.721 # [m/s2]
# rho_atm = 1.6e-2 # [kg/m3] mars atmospheric density

def d_dvar(arr: np.ndarray, pos_tuple: tuple, var_df: str, distance: float):
    j, i = pos_tuple

    if j + 1 >= 35:
        j = 0
    if i + 1 >= 60:
        i = -1
    
    if var_df == "dx":
        finite_diff = arr[j, i + 1] - arr[j, i - 1]
        finite_diff /= (2*distance)
    
    if var_df == "dy":
        finite_diff = arr[j - 1, i] - arr[j + 1, i] # Signos invertidos porque el indice del array aumenta mientra se baja
        finite_diff /= (2*distance)
    
    return finite_diff



def particles_speed(rho, vw, r, m):
	cs = np.pi * r**2 * rho / m
	return vw - np.power(grav, 1/2)/np.power(cs, 1/2)



def g(t, y, a, k):
    return a.dot(y) - np.array([0, -k])


def latitude_fullcircle():
    R = 3389.5e3 # Radio de Marte en metros: 3 389 500 m
    phi = np.linspace(5*np.pi/180, 175*np.pi/180, 35) # 35 partes porque asi vienen los datos
    c_list = []
    for angle in phi:
        radii = R*np.sin(angle)
        circ = 2*np.pi*radii
        repeat_circ = np.repeat(circ/360, 60)
        c_list.append(repeat_circ)

    return np.array(c_list)



def longitude_halfcircle():
    R = 3389.5e3 # Radio de Marte en metros: 3 389 500 m
    halfcircle = np.pi*R # Media circunferencia 2*pi*R / 2
    halfcircle /= 180
    return np.array(halfcircle)



def compute_volume(lat_m, lon_m, zagl_m):
    return lat_m * lon_m * zagl_m



def compute_r0(radii_eff, var_eff):
    return radii_eff/(np.power(1 + var_eff, 5/2))


def compute_sigma0(var_eff):
    return np.sqrt(np.log(1 + var_eff))


def particles_density(n_part, vol):
    return n_part/vol


def grain_mass(radi):
    mass = 4/3 * np.pi * rhodust * radi**3
    return mass


def dq_eq(r_small):
    dq = (1e-15/1e-6) * r_small
    return dq


def dq_nq(dphi, r_small, r_large):
    e_charge = 1.60e-19 # Carga del electron [C]
    rf = (r_small * r_large)/(r_small + r_large)
    dq = 2668 * (dphi/2) * (rf/0.5e-6) * e_charge
    return dq


def compute_rmean(mass_dust, N0, sigma):
    num = 3 * mass_dust
    den = 4 * np.pi * rhodust * N0
    return ( num/den )**(1/3) * np.exp(-1.5 * sigma**2)


def dist_division(points, r_mean, sigma):
    r_std = r_mean + 0.25*r_mean*sigma
    l_std = r_mean - 0.25*r_mean*sigma

    l_points = points < l_std
    r_points = points > r_std

    return  points[l_points], points[r_points], l_points, r_points


def grain_log_normal(points, r0, sigma0):
    # [Madeleine, 2011] eq. 2.2, 2.6
    
    # sigma0 = compute_sigma0(var_eff) # La variance effectif
    # r0     = compute_rmean(mass_dust, N0, sigma0) # Le rayon effectif

    den = np.sqrt(2 * np.pi) * sigma0 * points
    exp_arg = -0.5 * ( np.log(points/r0) / sigma0)**2

    return 1/den * np.exp(exp_arg)