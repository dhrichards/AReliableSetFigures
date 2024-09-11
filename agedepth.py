#%%

import numpy as np
from scipy.interpolate import interp1d,griddata
import pandas as pd



data = pd.read_excel("./data/UpstreamCorrection_data_10Jun2021.xls",
                     sheet_name=4)





depth = data['depth'].to_numpy()[2:].astype(float)
age = data['age'].to_numpy()[2:].astype(float)
dist = data['upstream distance'].to_numpy()[2:].astype(float)*1000


def time2depth(t):
    return np.interp(t,age,depth)



def dist2depth(s):
    return np.interp(s,dist,depth)

def depth2time(d):
    return np.interp(d,depth,age)

def depth2dist(d):
    return np.interp(d,depth,dist)


def depth2time(d):
    return np.interp(d,depth,age)
