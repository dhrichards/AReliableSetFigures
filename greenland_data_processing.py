#%%
import rasterio
import numpy as np
import netCDF4 as nc
import xarray as xr
from SavitzkyGolay import sgolay2d

#%%


def save_bedmachine_var_to_geotiff(var_name, bedmachine_file):
    with nc.Dataset(bedmachine_file) as ds:
        data = np.squeeze(ds[var_name][:])
        x = np.squeeze(ds['x'][:])
        y = np.squeeze(ds['y'][:])
        y, data = np.flip(y, 0), np.flip(data, 0)

        x_resolution = abs(x[1] - x[0])
        y_resolution = abs(y[1] - y[0])

        transform = rasterio.transform.from_origin(x[0], y[0], x_resolution, y_resolution)
        
        crs_proj = rasterio.crs.CRS.from_epsg(3413)  # WGS 84 / NSIDC Sea Ice Polar Stereographic North projection

        with rasterio.open(f'{var_name}.tif', 'w',
                           driver='GTiff',
                           height=data.shape[0],
                           width=data.shape[1],
                           count=1,
                           dtype=data.dtype,
                           crs=crs_proj,
                           transform=transform) as dst:
            dst.write(data, 1)



def apply_filter_and_save(input_file, window=51, order=5):

    #subtract .tif from input_file
    output_file = input_file[:-4]
    with rasterio.open(input_file) as src:
        data = src.read(1)
        
        # Apply the Savitzky-Golay filter
        data_filtered = sgolay2d(data, window, order)
        d_data_x, d_data_y = sgolay2d(data, window, order, 'both')
        d_data_x /= src.res[0]
        d_data_y /= src.res[0]

        # Save the filtered data and its derivatives as separate GeoTIFF files
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1)

        with rasterio.open(f"{output_file}_filtered.tif", 'w', **profile) as dst:
            dst.write(data_filtered.astype(rasterio.float32), 1)

        with rasterio.open(f"{output_file}_dx_filtered.tif", 'w', **profile) as dst:
            dst.write(d_data_x.astype(rasterio.float32), 1)

        with rasterio.open(f"{output_file}_dy_filtered.tif", 'w', **profile) as dst:
            dst.write(d_data_y.astype(rasterio.float32), 1)




class LocalBilinearInterpolator:
    def __init__(self, filename):
        self.filename = filename
        self.band = 1

        with rasterio.open(self.filename) as src:
            self.x, _ = src.xy(0, np.arange(src.width))
            _, self.y = src.xy(np.arange(src.height), 0)

    def interpolate(self, xi, yi):
        with rasterio.open(self.filename) as src:
            
            x1i = np.searchsorted(self.x, xi, side='left') - 1
            x2i = np.searchsorted(self.x, xi, side='right')
            y1i = self.y.size - np.searchsorted(self.y[::-1], yi, side='right') + 1
            y2i = self.y.size - np.searchsorted(self.y[::-1], yi, side='left')

            # Load the data for the four nearest points
            Q11 = src.read(self.band, window=rasterio.windows.Window(x1i, y1i, 1, 1)).squeeze()
            Q12 = src.read(self.band, window=rasterio.windows.Window(x1i, y2i, 1, 1)).squeeze()
            Q21 = src.read(self.band, window=rasterio.windows.Window(x2i, y1i, 1, 1)).squeeze()
            Q22 = src.read(self.band, window=rasterio.windows.Window(x2i, y2i, 1, 1)).squeeze()
            x1 = self.x[x1i]
            x2 = self.x[x2i]
            y1 = self.y[y1i]
            y2 = self.y[y2i]

            # Perform bilinear interpolation
            fxy1 = Q11 * (x2 - xi) / (x2 - x1) + Q21 * (xi - x1) / (x2 - x1)
            fxy2 = Q12 * (x2 - xi) / (x2 - x1) + Q22 * (xi - x1) / (x2 - x1)
            data = fxy1 * (y2 - yi) / (y2 - y1) + fxy2 * (yi - y1) / (y2 - y1)

            # # Calculate gradients
            # d_data_dx = ((Q21 - Q11) / (x2 - x1)) * (y2 - yi) / (y2 - y1) + ((Q22 - Q12) / (x2 - x1)) * (yi - y1) / (y2 - y1)
            # d_data_dy = ((Q12 - Q11) / (y2 - y1)) * (x2 - xi) / (x2 - x1) + ((Q22 - Q21) / (y2 - y1)) * (xi - x1) / (x2 - x1)

            return data


def query_point(x,y):
    interp_vx = LocalBilinearInterpolator(\
        './data/greenland_vel_mosaic250_vx_v1_filtered.tif')
    
    interp_vy = LocalBilinearInterpolator(\
        './data/greenland_vel_mosaic250_vy_v1_filtered.tif')
    
    interp_dudx = LocalBilinearInterpolator(\
        './data/greenland_vel_mosaic250_vx_v1_dx_filtered.tif')
    interp_dudy = LocalBilinearInterpolator(\
        './data/greenland_vel_mosaic250_vx_v1_dy_filtered.tif')
    interp_dvdx = LocalBilinearInterpolator(\
        './data/greenland_vel_mosaic250_vy_v1_dx_filtered.tif')
    interp_dvdy = LocalBilinearInterpolator(\
        './data/greenland_vel_mosaic250_vy_v1_dy_filtered.tif')

    interp_bed = LocalBilinearInterpolator('./data/bed_filtered.tif')
    interp_surf = LocalBilinearInterpolator('./data/surface_filtered.tif')
    interp_thickness = LocalBilinearInterpolator('./data/thickness_filtered.tif')

    interp_dbeddx = LocalBilinearInterpolator('./data/bed_dx_filtered.tif')
    interp_dbeddy = LocalBilinearInterpolator('./data/bed_dy_filtered.tif')
    interp_dsurfdx = LocalBilinearInterpolator('./data/surface_dx_filtered.tif')
    interp_dsurfdy = LocalBilinearInterpolator('./data/surface_dy_filtered.tif')

    class surface_point:
        def __init__(self,x,y):
            self.x = x
            self.y = y
            self.vx = interp_vx.interpolate(x,y)
            self.vy = interp_vy.interpolate(x,y)
            self.gradu = np.array([[interp_dudx.interpolate(x,y),interp_dudy.interpolate(x,y)],\
                                   [interp_dvdx.interpolate(x,y),interp_dvdy.interpolate(x,y)]])
            self.bed = interp_bed.interpolate(x,y)
            self.surf = interp_surf.interpolate(x,y)
            self.thickness = interp_thickness.interpolate(x,y)
            self.dbeddx = interp_dbeddx.interpolate(x,y)
            self.dbeddy = interp_dbeddy.interpolate(x,y)
            self.dsurfdx = interp_dsurfdx.interpolate(x,y)
            self.dsurfdy = interp_dsurfdy.interpolate(x,y)



    
    return surface_point(x,y)
    


def path2d(time,dt,xc,yc):
    
    t = np.arange(0,time,dt)
    nt = t.size
    gradu = np.zeros((nt,3,3))
    xp=np.zeros(nt)
    yp = np.zeros(nt)
    t=np.zeros(nt)
    s=np.zeros(nt)
    vx = np.zeros(nt)
    vy = np.zeros(nt)
    bed = np.zeros(nt)
    surf = np.zeros(nt)
    thickness = np.zeros(nt)
    dbeddx = np.zeros(nt)
    dbeddy = np.zeros(nt)
    dsurfdx = np.zeros(nt)
    dsurfdy = np.zeros(nt)
        
    xp[-1]=xc
    yp[-1]=yc
    dt=dt

    interp_vx = LocalBilinearInterpolator(\
        './data/greenland_vel_mosaic250_vx_v1_filtered.tif')
    
    interp_vy = LocalBilinearInterpolator(\
        './data/greenland_vel_mosaic250_vy_v1_filtered.tif')
    
    interp_dudx = LocalBilinearInterpolator(\
        './data/greenland_vel_mosaic250_vx_v1_dx_filtered.tif')
    interp_dudy = LocalBilinearInterpolator(\
        './data/greenland_vel_mosaic250_vx_v1_dy_filtered.tif')
    interp_dvdx = LocalBilinearInterpolator(\
        './data/greenland_vel_mosaic250_vy_v1_dx_filtered.tif')
    interp_dvdy = LocalBilinearInterpolator(\
        './data/greenland_vel_mosaic250_vy_v1_dy_filtered.tif')

    interp_bed = LocalBilinearInterpolator('./data/bed_filtered.tif')
    interp_surf = LocalBilinearInterpolator('./data/surface_filtered.tif')
    interp_thickness = LocalBilinearInterpolator('./data/thickness_filtered.tif')

    interp_dbeddx = LocalBilinearInterpolator('./data/bed_dx_filtered.tif')
    interp_dbeddy = LocalBilinearInterpolator('./data/bed_dy_filtered.tif')
    interp_dsurfdx = LocalBilinearInterpolator('./data/surface_dx_filtered.tif')
    interp_dsurfdy = LocalBilinearInterpolator('./data/surface_dy_filtered.tif')

    
    for i in range(nt-1,0,-1):



        vx[i] = interp_vx.interpolate(xp[i],yp[i])
        vy[i] = interp_vy.interpolate(xp[i],yp[i])
        gradu[i,0,0] = interp_dudx.interpolate(xp[i],yp[i])
        gradu[i,0,1] = -interp_dudy.interpolate(xp[i],yp[i])
        gradu[i,1,0] = interp_dvdx.interpolate(xp[i],yp[i])
        gradu[i,1,1] = -interp_dvdy.interpolate(xp[i],yp[i])



        if i==nt-1:
            xp[i-1] = xp[i] - vx[i]*dt
            yp[i-1] = yp[i] - vy[i]*dt
        else:
            xp[i-1] = xp[i] - 1.5*vx[i]*dt + 0.5*vx[i-1]*dt
            yp[i-1] = yp[i] - 1.5*vy[i]*dt + 0.5*vy[i-1]*dt
        
        t[i-1] = t[i] - dt
        ds = np.sqrt((xp[i-1]-xp[i])**2 + (yp[i-1]-yp[i])**2)
        s[i-1] = s[i] - ds



    vx[0]= interp_vx.interpolate(xp[0], yp[0])
    vy[0]= interp_vy.interpolate(xp[0], yp[0])
    gradu[0,0,0] = interp_dudx.interpolate(xp[0], yp[0])
    gradu[0,0,1] = -interp_dudy.interpolate(xp[0], yp[0])
    gradu[0,1,0] = interp_dvdx.interpolate(xp[0], yp[0])
    gradu[0,1,1] = -interp_dvdy.interpolate(xp[0], yp[0])


    for i in range(nt):
        bed[i]= interp_bed.interpolate(xp[i],yp[i])
        surf[i] = interp_surf.interpolate(xp[i],yp[i])
        thickness[i] = interp_thickness.interpolate(xp[i],yp[i])
        dbeddx[i] = interp_dbeddx.interpolate(xp[i],yp[i])
        dbeddy[i] = interp_dbeddy.interpolate(xp[i],yp[i])
        dsurfdx[i] = interp_dsurfdx.interpolate(xp[i],yp[i])
        dsurfdy[i] = interp_dsurfdy.interpolate(xp[i],yp[i])

    path_dict = {'xp':xp, 'yp':yp, 't':t, 's':s, 'vx':vx,\
                'vy':vy, 'bed':bed, 'surf':surf, \
                'thickness':thickness, 'dbeddx':dbeddx, \
                'dbeddy':dbeddy, 'dsurfdx':dsurfdx,\
                'dsurfdy':dsurfdy, 'gradu':gradu, 'dt':dt, 'nt':nt}
    

    return path_dict


    



#%%