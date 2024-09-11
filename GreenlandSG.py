#%%

import rasterio
import netCDF4 as nc
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from SavitzkyGolay import sgolay2d

fp = r'./greenland_vel_mosaic250_vx_v1.tif'
img = rasterio.open(fp)
vx = img.read(1)

fp = r'./greenland_vel_mosaic250_vy_v1.tif'
img = rasterio.open(fp)
vy = img.read(1)


y = np.array(img.xy(np.arange(vy.shape[0]),np.zeros(vy.shape[0]))[1])
x = np.array(img.xy(np.zeros(vy.shape[1]),np.arange(vy.shape[1]))[0])




##Add nans
vx[vx<-1e9]=np.nan
vy[vy<-1e9]=np.nan

#flip on axis 0 to makes postitive
vx = np.flip(vx,0)
vy = np.flip(vy,0)
y = np.flip(y,0)

#Trim
# xmin = 100000
# xmax = 295000
# ymin = -2260000
# ymax = -1040000
xmin = 110000
xmax = 310000
ymin = -1940000
ymax = -1500000


xminind = np.abs(x-xmin).argmin()
xmaxind = np.abs(x-xmax).argmin()
yminind = np.abs(y-ymin).argmin()
ymaxind = np.abs(y-ymax).argmin()

x = x[xminind:xmaxind]
y = y[yminind:ymaxind]
vx = vx[yminind:ymaxind,xminind:xmaxind]
vy = vy[yminind:ymaxind,xminind:xmaxind]


#Savitzky-Golay filtering
window=51
order =5
vx_filt = sgolay2d(vx,window,order)
vy_filt = sgolay2d(vy,window,order)

dudx,dudy = sgolay2d(vx,window,order,'both')/(x[1]-x[0])
dvdx,dvdy = sgolay2d(vy,window,order,'both')/(y[1]-y[0])


fn = 'BedMachineGreenland-v5.nc'
ds = nc.Dataset(fn)


surface=np.squeeze(ds['surface'][:])
thickness=np.squeeze(ds['thickness'][:])
bed=np.squeeze(ds['bed'][:])
y2 = np.squeeze(ds['y'][:])
x2 = np.squeeze(ds['x'][:])

y2=np.flip(y2,0)
surface = np.flip(surface,0)
thickness = np.flip(thickness,0)
bed = np.flip(bed,0)

window=51
order =5
dbeddx,dbeddy = sgolay2d(bed,window,order,'both')/(x[1]-x[0])
dsurfdx,dsurfdy = sgolay2d(surface,window,order,'both')/(y[1]-y[0])


fn = 'Greenland1km.nc'
ds = nc.Dataset(fn)
acc = np.squeeze(ds['presprcp'][:])
temp = np.squeeze(ds['presartm'][:])
y3 = np.squeeze(ds['y'][:])
x3 = np.squeeze(ds['x'][:])






class gl:
    def __init__(self):
        self.vx_interp = RegularGridInterpolator((x,y), vx.T)
        self.vy_interp = RegularGridInterpolator((x,y), vy.T)
        self.dudx_interp = RegularGridInterpolator((x,y), dudx.T)
        self.dudy_interp = RegularGridInterpolator((x,y), dudy.T)
        self.dvdx_interp = RegularGridInterpolator((x,y), dvdx.T)
        self.dvdy_interp = RegularGridInterpolator((x,y), dvdy.T)
        self.surf_interp = RegularGridInterpolator((x2,y2), surface.T)
        self.thick_interp = RegularGridInterpolator((x2,y2), thickness.T)
        self.bed_interp = RegularGridInterpolator((x2,y2), bed.T)
        self.accumulation_interp = RegularGridInterpolator((x3,y3), acc.T)
        self.surftemp_interp = RegularGridInterpolator((x3,y3), temp.T)
        self.dbeddx_interp = RegularGridInterpolator((x2,y2),dbeddx.T)
        self.dbeddy_interp = RegularGridInterpolator((x2,y2),dbeddy.T)
        self.dsurfdx_interp = RegularGridInterpolator((x2,y2), dsurfdx.T)
        self.dsurfdy_interp = RegularGridInterpolator((x2,y2), dsurfdy.T)
        
        self.xmin = np.max([x.min(),x2.min(),x3.min()])
        self.xmax = np.min([x.max(),x2.max(),x3.max()])
        self.ymin = np.max([y.min(),y2.min(),y3.min()])
        self.ymax = np.min([y.max(),y2.max(),y3.max()])
        
                
    def interps(self,xi,yi):
        self.xi=xi
        self.yi=yi
        
        if xi.ndim==2:
            pts = np.reshape([xi,yi],(2,-1)).T
        elif xi.ndim==1:
            pts = np.stack((xi,yi),1)
        else:
            pts=[xi,yi]

        self.vx=self.vx_interp(pts)
        self.vy=self.vy_interp(pts)
        self.dudx = self.dudx_interp(pts)
        self.dudy = self.dudy_interp(pts)
        self.dvdx = self.dvdx_interp(pts)
        self.dvdy = self.dvdy_interp(pts)
        self.surface = self.surf_interp(pts)
        self.thickness = self.thick_interp(pts)
        self.bed = self.bed_interp(pts)
        self.accumulation = self.accumulation_interp(pts)
        self.surftemp = self.surftemp_interp(pts)

        self.surftemp = self.surftemp_interp(pts)
        self.dbeddx = self.dbeddx_interp(pts)
        self.dbeddy = self.dbeddy_interp(pts)
        self.dsurfdx = self.dsurfdx_interp(pts)
        self.dsurfdy = self.dsurfdy_interp(pts)
        if xi.ndim==2:
            self.vx=self.vx.reshape(xi.shape)
            self.vy=self.vy.reshape(xi.shape)
            self.dudx=self.dudx.reshape(xi.shape)
            self.dudy=self.dudy.reshape(xi.shape)
            self.dvdx=self.dvdx.reshape(xi.shape)
            self.dvdy=self.dvdy.reshape(xi.shape)
            self.surface=self.surface.reshape(xi.shape)
            self.thickness=self.thickness.reshape(xi.shape)
            self.bed=self.bed.reshape(xi.shape)
            self.accumulation=self.accumulation.reshape(xi.shape)
            self.surftemp=self.surftemp.reshape(xi.shape)
            self.dbeddx=self.dbeddx.reshape(xi.shape)
            self.dbeddy=self.dbeddy.reshape(xi.shape)
            self.dsurfdx=self.dsurfdx.reshape(xi.shape)
            self.dsurfdy=self.dsurfdy.reshape(xi.shape)

        self.vmag=np.sqrt(self.vx**2+self.vy**2)
        



# %%