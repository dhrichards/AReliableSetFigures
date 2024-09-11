
import numpy as np
from scipy.signal import savgol_filter
import agedepth
from scipy.optimize import minimize_scalar
import divide_data
import densityfromdepth
from greenland_data_processing import query_point
import scipy.integrate as integrate
import mcfab as mc_fab
# import specfabwrapper as sfw
import jax.numpy as jnp
import mcfabnp as mcnp


def fabric(path,npoints=5000,x='Richards',model=0,solver='mc',tind=0,fabric0=None):

    if isinstance(x,str):
        if x == 'Richards':
            x_tile = mc_fab.parameters.Richards2021(path.T[tind:])
        elif x == 'Elmer':
            x_tile = mc_fab.parameters.Elmer(path.T[tind:])
    else:    
        x_tile = np.tile(x,(path.gradu[tind:].shape[0],1))

    

    dt_tile = np.tile(path.dt,(path.gradu[tind:].shape[0],))


    if solver=='mc':
        n,m,a2,a4 = mc_fab.mcsolver.solve(npoints,path.gradu[tind:],
                                        dt_tile,x_tile,model,
                                        fabric0=fabric0)
        fabric = a2,a4,(n[-1,...],m[-1,...])
    elif solver=='a2':
        a2,a4 = mc_fab.a2evolution.solve(path.gradu[tind:],dt_tile,x_tile,model,closure=1)
        fabric = a2,a4,0

    elif solver=='mcnp':
        n,m,a2,a4 = mcnp.solve(npoints,path.gradu[tind:],
                                        dt_tile,x_tile,
                                        fabric0=fabric0)
        fabric = a2,a4,(n[-1,...],m[-1,...])

    # else:
    #     L = npoints
    #     nlm,a2,a4 = sfw.solve(L,path.gradu,dt_tile,x_tile,model)
    #     fabric = a2,a4,nlm

    return fabric

# def fabrics(paths,npoints=5000,x,sr_type='SR',model='Sachs',solver='mc'):
#     # convert paths into jax arrays and vmap fabric function




def streampaths(depths,path2d):
    times=-agedepth.depth2time(depths)


    paths=[]
    for t in times:
        nt = path2d['t'].size - np.abs(path2d['t'] - t).argmin()
        paths.append(stream(path2d,nt))


    for p in paths:
        p.optimizeacc()
        p.Temperature(-30,-30)

    return paths

    

class stream:
    def __init__(self,path2d,nt):
        self.nt = nt
        self.t = path2d['t'][-nt:,...]
        self.dt = path2d['t'][1]-path2d['t'][0]

        self.gradu = path2d['gradu'][-nt:,...]
        #dw/dz from div u = 0 - inital guess not accounting for density change in firn
        self.gradu[:,2,2] = - self.gradu[:,0,0] - self.gradu[:,1,1]

        self.xp = path2d['xp'][-nt:,...]
        self.yp = path2d['yp'][-nt:,...]
        self.s = path2d['s'][-nt:,...]


        surf_smooth = savgol_filter(path2d['surf'],211,3)
        surfslope = np.gradient(surf_smooth,path2d['s'])
        self.surfslope = surfslope[-nt:,...]
        self.meanslope = np.mean(surfslope)

        self.vx_s = path2d['vx'][-nt:,...] # note these a surface velocities
        self.vy_s = path2d['vy'][-nt:,...]

        self.bed = path2d['bed'][-nt:,...]
        #self.acc = path2d['acc_gl[-nt:,...]
        self.surf = path2d['surf'][-nt:,...]
        # self.dbeddx = path2d['dbeddx'][-nt:,...]
        # self.dbeddy = path2d['dbeddy'][-nt:,...]
        # self.dsurfdx = path2d['dsurfdx'][-nt:,...]
        # self.dsurfdy = path2d['dsurfdy'][-nt:,...]
        #self.basalmelt = path2d['basalmelt[-nt:,...]


    def optimizeacc(self): #optimize accumulation rate to match depth age relationship from gerber
        res = minimize_scalar(self.deptherror,method='bounded',bounds=(0,1000))
        self.acc = res.x
        self.depth(self.acc)



    def deptherror(self,acc): #error function for optimization
        self.depth(acc)
        depth = self.d[-1]

        error = np.sqrt((depth - agedepth.time2depth(-self.t[0]))**2)

        return error
    

    def depth(self,acc=0.1): #calculate depth of path using accumulation rate

                
        self.z=np.zeros(self.nt)
        self.vz=np.zeros(self.nt)
        self.d=np.zeros(self.nt)
        self.density=np.zeros(self.nt)

        



        vs = np.sqrt(self.vx_s**2 + self.vy_s**2)

        # Averaging to remove noise which can cause crossing streamlines due to
        # fluctuations in vertical velocity


        # assign initial values to vertical velocity and depth
        self.vz[0]= -acc + vs[0]*self.surfslope[-1]
        self.z[0]=self.surf[0]
        self.d[0] = 0
        self.density[0] = densityfromdepth.densityfromdepth(self.d[0])

        
        for i in range(self.nt-1):


            self.z[i+1] = self.z[i] + self.vz[i]*self.dt
            self.d[i+1] = self.surf[i+1]-self.z[i+1]
            self.density[i+1] = densityfromdepth.densityfromdepth(self.d[i+1])
            drhodt = (self.density[i+1]-self.density[i])/self.dt
            
            if i==0: # forward difference drhodt for first timestep
                self.gradu[i,2,2] = -self.gradu[i,0,0] - self.gradu[i,1,1]\
                          #- drhodt/self.density[i]
                

            self.gradu[i+1,2,2] = -self.gradu[i+1,0,0] - self.gradu[i+1,1,1]\
                      #- drhodt/self.density[i+1]

            self.vz[i+1] = self.vz[i]/(1-self.gradu[i,2,2]*self.dt) # check this
            
            
        
        self.ztilde = (self.z-self.bed)/(self.surf-self.bed)       
        self.dtilde = 1-self.ztilde


    def Temperature(self,location='EGRIP',Ts=-30,Tb=-5):
        #Use grip data for temperature, renormalize to depth at GRIP
        if location == 'EGRIP':
            import divide_data
            data = divide_data.data('GRIP')
            self.T = data.Temperature(self.dtilde*data.H)
        elif location == 'HWD2':
            import shelf_data
            data = shelf_data.data('HWD2')
            self.T = data.Temperature(self.dtilde*data.H)
        # self.T = Ts + (Tb-Ts)*self.dtilde
        # self.Ts = Ts
        # self.Tb = Tb
            


    



class divide:
    def __init__(self,dh=0.8,dt=1,location='GRIP',model='DJ',include_shear=False):

        self.dt = dt
        nt_max = 100000
        self.nt = nt_max
        self.dh = dh

        self.location = location
        self.include_shear = include_shear

        self.model = model

        self.t = np.arange(0,nt_max*dt,dt)

        self.gradu = np.zeros((nt_max,3,3))
        self.vz = np.zeros(nt_max)
        
        self.z = np.zeros(nt_max)
        self.d = np.zeros(nt_max)
        self.density = np.zeros(nt_max)

        self.data = divide_data.data(location)
        self.H = self.data.H
        self.acc = self.data.acc

        if include_shear:
            self.surface = query_point(self.data.xc,self.data.yc)

        


        

        
    def D_zz(self,depth):

        if self.model=='DJ':
            d_zz = self.Dansgard_Johnson(depth)
        elif self.model == 'Nye':
            d_zz = self.Nye(depth)

        return d_zz


    def Dansgard_Johnson(self,depth):
        #d_switch = 1750 #m, depth above which D_zz is constant, copying Castelnau (1996)
        d_switch = 0.66666*self.H
        # Change above
        # Integration of dansgard johnson profile to transition from vz_0 to 0 at base
        # vz_0 = -acc = int dw/dz dz = dzz_0*d_switch + 1/2*dzz_0*(H-d_switch)**2
        D_zz_0 = -self.acc/(d_switch + 0.5*(self.H-d_switch))

        depths = np.array([0,d_switch,self.H])
        d_zzs = np.array([D_zz_0,D_zz_0,0])

        return np.interp(depth,depths,d_zzs)

    def Nye(self,depth):
        d_zz = -self.acc/self.H
        return d_zz*np.ones_like(depth)

    def Temperature(self,d):
        return self.data.Temperature(d)


        
    def depth(self):

        self.z[0] = self.H
        self.vz[0] = -self.acc
        self.strain = np.zeros(self.nt)

        self.gradu[0,2,2] = self.D_zz(self.d[0])
        if self.include_shear:
            k = -self.gradu[0,2,2]/(self.surface.gradu[0,0] + self.surface.gradu[1,1])
            self.gradu[0,0:2,0:2] = k*self.surface.gradu[0:2,0:2]
        else:
            k = 0.5
            self.gradu[0,0,0] = self.gradu[0,1,1] = -self.gradu[0,2,2]/2
        
        i=0
        while self.d[i] < self.dh*self.H:
            self.z[i+1] = self.z[i] + self.vz[i]*self.dt
            self.d[i+1] = self.H-self.z[i+1]
            self.strain[i+1] = self.strain[i] + self.gradu[i,2,2]*self.dt
            
            self.gradu[i+1,2,2] = self.D_zz(self.d[i+1])
            self.gradu[i+1,0:2,0:2] = self.gradu[0,0:2,0:2]*self.gradu[i+1,2,2]/self.gradu[0,2,2]
            
            
            #self.vz[i+1] = self.vz[i]/(1-self.gradu[i,2,2]*self.dt) # check this
            self.vz[i+1] = self.vz[i] + self.gradu[i,2,2]*self.dt*self.vz[i] # check this
            i+=1

        self.nt = i
        self.z = self.z[:self.nt]
        self.d = self.d[:self.nt]
        self.vz = self.vz[:self.nt]
        self.strain = self.strain[:self.nt]
        self.gradu = self.gradu[:self.nt,...]
        self.t = self.t[:self.nt]
        self.dtilde = self.d/self.H

        self.T = self.data.Temperature(self.d)

        if self.include_shear:
            self.verticalderivativesSIA()
    

    def verticalderivativesSIA(self,n=3):
        #for n=3 dudz = 4Ud**3/t where d = 1-ztilde
        
        self.ux = self.verticalprofile(self.surface.vx)
        self.uy = self.verticalprofile(self.surface.vy)

        self.gradu[:,0,2] = np.gradient(self.ux,self.z)
        self.gradu[:,1,2] = np.gradient(self.uy,self.z)


    def verticalprofile(self,u_s,n=3):
        '''For variable temperature with depth
        u_z = u_s * int_0^d A(d)d^n dd /
                    int_0^H A(d)d^n dd'''
        
        dtemp = np.linspace(0,self.H,1000)
        Ttemp = self.Temperature(dtemp)

        
        int_part = integrate.cumulative_trapezoid(self.A(self.T)*self.d**n,self.d,initial=0)
        int_full = integrate.trapezoid(self.A(Ttemp)*dtemp**n,dtemp)

        return u_s*int_part/int_full

    def A(self,T):
        secperyear = 365.25*24*60*60
        T = T + 273.15
        A = np.zeros_like(T)
        A0_l = 3.985e-13*secperyear
        A0_h = 1.916e3*secperyear

        Q_l = 60e3
        Q_h = 139e3

        R = 8.314

        A[T<263.15] = A0_l*np.exp(-Q_l/(R*T[T<263.15]))
        A[T>=263.15] = A0_h*np.exp(-Q_h/(R*T[T>=263.15]))

        return A







class sia:
    def __init__(self,nt,location):
        from greenland_data_processing import  query_point
        self.data = divide_data.data(location)

        self.surface_point = query_point(self.data.xc,self.data.yc)

        self.nt = nt




        self.gradu = path2d.gradu[-nt:,...]
        #dw/dz from div u = 0 - inital guess not accounting for density change in firn
        self.gradu[:,2,2] = - self.gradu[:,0,0] - self.gradu[:,1,1]

        self.xp = path2d.xp[-nt:,...]
        self.yp = path2d.yp[-nt:,...]
        self.s = path2d.s[-nt:,...]


        surf_smooth = savgol_filter(path2d.surf,211,3)
        surfslope = np.gradient(surf_smooth,path2d.s)
        self.surfslope = surfslope[-nt:,...]
        self.meanslope = np.mean(surfslope)

        self.vx_s = path2d.vx[-nt:,...] # note these a surface velocities
        self.vy_s = path2d.vy[-nt:,...]

        self.bed = path2d.bed[-nt:,...]
        #self.acc = path2d.acc_gl[-nt:,...]
        self.surf = path2d.surf[-nt:,...]
        self.dbeddx = path2d.dbeddx[-nt:,...]
        self.dbeddy = path2d.dbeddy[-nt:,...]
        self.dsurfdx = path2d.dsurfdx[-nt:,...]
        self.dsurfdy = path2d.dsurfdy[-nt:,...]
        #self.basalmelt = path2d.basalmelt[-nt:,...]



        
    def verticalprofile(self,ztilde,n=3):
        return 1 - (1-ztilde)**(n+1)

    

    def depth(self,n=3):


        self.z=np.zeros(self.nt)
        self.vz=np.zeros(self.nt)
        self.ztilde = np.zeros(self.nt)
        self.dt = np.zeros(self.nt)

        #surfaceslope = (self.surf[100]-self.surf[0])/(self.s[100]-self.s[0])

        #surf_smooth = savgol_filter(self.surf,211,3)
        #surfslope = np.gradient(surf_smooth,self.s)

        vs = np.sqrt(self.vx_s**2 + self.vy_s**2)

        vz_s = -0.1+ vs*np.mean(self.surfslope) 
        # doing this to remove noise which can cause crossing streamlines due to
        # fluctuations in vertical velocity - 0.132 is average accumulation along whole path

        self.vz[0]=vz_s[0]
        self.z[0]=self.surf[0]


        #Set to give correct total change - to be later multiplied by vertical profile
        #self.gradu[:,2,2] = ((vz_s)/(self.surf-self.bed))/(1-self.k/(n+2))


        
        
        for i in range(self.nt-1):
            self.ztilde[i] = (self.z[i]-self.bed[i])/(self.surf[i]-self.bed[i])

            

            #Scale surface velocity gradients relative to depth
            self.gradu[i,...] = self.gradu[i,...]*self.verticalprofile(self.ztilde[i],n)

           
            # Increase timestep as depth increases so we remain at same 
            # x and y position even as du/dx etc decreases
            self.dt[i] = (self.t[i+1]-self.t[i])/self.verticalprofile(self.ztilde[i],n)


            #self.vz[i+1] = self.vz[i] + self.gradu[i,2,2]*self.vz[i]*self.dt[i]
            self.vz[i+1] = self.vz[i]/(1-self.gradu[i,2,2]*self.dt[i])
            self.z[i+1] = self.z[i] + self.vz[i]*self.dt[i]
            self.t[i+1] = self.t[i] + self.dt[i] 

        
        self.d=self.surf-self.z       
        self.ztilde = (self.z-self.bed)/(self.surf-self.bed)

        # Add extra part to derivatives
        dtilde = 1-self.ztilde

        self.gradu[:,0,0] = self.gradu[:,0,0] + self.vx_s*self.k*(n+1)*dtilde**n*\
            (self.z*(self.dsurfdx-self.dbeddx) + self.dbeddx*self.surf - self.dsurfdx*self.bed)/\
            (self.surf-self.bed)**2

        self.gradu[:,0,1] = self.gradu[:,0,1] + self.vx_s*self.k*(n+1)*dtilde**n*\
            (self.z*(self.dsurfdy-self.dbeddy) + self.dbeddy*self.surf - self.dsurfdy*self.bed)/\
            (self.surf-self.bed)**2

        self.gradu[:,1,0] = self.gradu[:,1,0] + self.vy_s*self.k*(n+1)*dtilde**n*\
            (self.z*(self.dsurfdx-self.dbeddx) + self.dbeddx*self.surf - self.dsurfdx*self.bed)/\
            (self.surf-self.bed)**2

        self.gradu[:,1,1] = self.gradu[:,1,1] + self.vy_s*self.k*(n+1)*dtilde**n*\
            (self.z*(self.dsurfdy-self.dbeddy) + self.dbeddy*self.surf - self.dsurfdy*self.bed)/\
            (self.surf-self.bed)**2

        #Or just continuity
        self.gradu[:,2,2] = - self.gradu[:,0,0] - self.gradu[:,1,1]
        



    def verticalderivativesSIA(self,n=3):

        #for n=3 dudz = 4Ud**3/t where d = 1-ztilde
        dtilde = 1-self.ztilde
        thick = self.surf - self.bed

        self.gradu[:,0,2] = (n+1)*self.vx_s*dtilde**n/thick
        self.gradu[:,1,2] = (n+1)*self.vy_s*dtilde**n/thick



        # def verticalderivatives(self):

        # #Time derviatives
        # dudt = np.gradient(self.vx,self.t)
        # dvdt = np.gradient(self.vy,self.t)

        # #Vertical gradients
        # self.gradu[:,0,2] = (dudt - self.vx*self.gradu[:,0,0] - self.vy*self.gradu[:,0,1])/self.vz
        # self.gradu[:,1,2] = (dvdt - self.vx*self.gradu[:,1,0] - self.vy*self.gradu[:,1,1])/self.vz

        # #Update D and W
        # self.D = 0.5*(self.gradu+np.swapaxes(self.gradu,1,2))
        # self.W = 0.5*(self.gradu-np.swapaxes(self.gradu,1,2))
