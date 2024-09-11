import numpy as np

data= np.genfromtxt('./data/EGRIP_eigenvalues.csv',delimiter='\t',skip_header=1)
def eigenvalues(dmin=0,dmax=10000):
    depth = data[:,4]


    eig1 = data[:,7]
    eig2 = data[:,9]
    eig3 = data[:,11]
    # eig3>eig2>eig1


    #Sort eigenvalues: below d=dswitch, e_n>e_z>e_s, above d=dwitch, e_z>e_n>e_s
    e_s = np.zeros_like(eig1)
    e_n = np.zeros_like(eig1)
    e_z = np.zeros_like(eig1)

    dswitch = 250
    e_n = np.where(depth>dswitch,eig3,eig2)
    e_z = np.where(depth>dswitch,eig2,eig3)
    e_s = eig1



    # Trim to depth range
    e_s = e_s[(depth>dmin) & (depth<dmax)]
    e_n = e_n[(depth>dmin) & (depth<dmax)]
    e_z = e_z[(depth>dmin) & (depth<dmax)]
    depth = depth[(depth>dmin) & (depth<dmax)]

    return depth,e_s,e_z,e_n


def curve_fit(dmin=0,dmax=10000):

    depth,e_s,e_z,e_n = eigenvalues(dmin=dmin,dmax=dmax)

    # Fit a curve to the data
    from scipy.optimize import curve_fit
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    popt, pcov = curve_fit(func, depth, e_s)
    e_s_fit = func(depth, *popt)

    popt, pcov = curve_fit(func, depth, e_n)
    e_n_fit = func(depth, *popt)

    popt, pcov = curve_fit(func, depth, e_z)
    e_z_fit = func(depth, *popt)

    return depth,e_s_fit,e_z_fit,e_n_fit


def rolling_average(dmin=0,dmax=10000,window=151):
    depth,e_s,e_z,e_n = eigenvalues(dmin=dmin,dmax=dmax)

    # Rolling average
    from scipy.signal import savgol_filter
    e_s_smooth = savgol_filter(e_s, window, 3)
    e_n_smooth = savgol_filter(e_n, window, 3)
    e_z_smooth = savgol_filter(e_z, window, 3)

    return depth,e_s_smooth,e_z_smooth,e_n_smooth


def interp(d,window=151):
    depth,e_s,e_z,e_n = rolling_average(window=window)

    e_s_interp = np.interp(d,depth,e_s,left=np.nan,right=np.nan)
    e_z_interp = np.interp(d,depth,e_z,left=np.nan,right=np.nan)
    e_n_interp = np.interp(d,depth,e_n,left=np.nan,right=np.nan)

    return e_s_interp,e_z_interp,e_n_interp






    



