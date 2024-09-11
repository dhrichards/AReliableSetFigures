import numpy as np
from scipy.interpolate import interp1d

#Digitised from Fig. 9 "Initial results from geophysical surveys and shallow coring of the
# Northeast Greenland Ice Stream (NEGIS)" by Vallelonga et al. (2014)"
rawdata = np.array([[0.3383916990920882, -0.35901626644076146],\
[0.4369649805447471, 5.930440910760732],\
[0.5783398184176395, 16.985123041666967],\
[0.6846952010376135, 30.20859729750667],\
[0.8040207522697794, 49.28549115454492],\
[0.8695201037613488, 63.75470137373827],\
[0.9019455252918288, 76.07611164841097],\
[0.9136186770428015, 86.5542085573124],\
[0.9149156939040207, 101.19735398564033]])

raw_d = rawdata[:,1]
raw_rho = rawdata[:,0]
raw_rho = raw_rho/raw_rho[-1]




def densityfromdepth(depth):
    # depth in m
    # normalised density where 1 = ice sheet density
    # from Fig. 9 "Initial results from geophysical surveys and shallow coring of the
    # Northeast Greenland Ice Stream (NEGIS)" by Vallelonga et al. (2014)
    # Digitised using https://apps.automeris.io/wpd/
    #
    f = interp1d(raw_d,raw_rho,kind='cubic',fill_value=1,bounds_error=False)
    density = f(depth)
    density[density>1]=1
    return density

