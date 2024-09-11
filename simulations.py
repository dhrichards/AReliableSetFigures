
import numpy as np
import matplotlib.pyplot as plt
import track
import divide_data
from tqdm import tqdm
import pickle
import agedepth
import stoll
import seaborn as sns
from mcfab import BuildHarmonicsSpecFab as BuildHarmonics
import cartopy.crs as ccrs
import mcfab as mc_fab
import jax.numpy as jnp




class simulation:
    def __init__(self,\
                 npoints,params,legend,model,\
                 colors = ['#1f77b4','#03045e', '#0077b6', '#00b4d8',\
                           '#90e0ef','#caf0f8','#f72585','#7209b7',\
                           '#3a0ca3','#4361ee','#4cc9f0'],title='',
                solver=None,fabric0=None,jit=None):
        self.params = params
        self.legend = legend
        self.model = model
    
        if solver is None:
            self.solver = ['mc']*len(params)
        else:
            self.solver = solver

        if jit is None:
            self.jit = [1]*len(params)
        else:
            self.jit = jit

        self.fabric0 = fabric0
        self.colors = colors
        self.npoints = npoints
        self.title = title


        
        self.scatter_kwargs = {'marker': 'o', \
                               'alpha': 0.5, 'color': self.colors[0],\
                                'edgecolors':'none'}



    def divide(self,nlocations=4,vertical=False,dt=100):
        locations = ['GRIP','Talos','DomeF','DomeC']
        locations = locations[:nlocations]

        # nlocs = 1 ,2, 3, 4 ncol = 1, 2, 3, 2 nrow = 1, 1, 1, 2
        ncol = int(np.ceil(np.sqrt(nlocations)))
        nrow = int(np.ceil(nlocations/ncol))


        width = ncol*4
        height = nrow*3
        fig,axs = plt.subplots(nrow,ncol,figsize=(width,height))

        for location in locations:
            if nlocations == 1:
                ax = axs
            else:
                ax = axs.flatten()[locations.index(location)]
            
            p = track.divide(dt=dt,location=location)
            p.depth()

            data =divide_data.data(location)


            if vertical:
                plot_data = (data.largest_ev, data.largest_ev_d)
            else:
                plot_data = (data.largest_ev_d, data.largest_ev)

            

            if location == locations[0]:
                ax.scatter(*plot_data,label='Ice cores',s=30,**self.scatter_kwargs)
            else:
                ax.scatter(*plot_data,s=30,**self.scatter_kwargs)
                
            for i in tqdm(range(len(self.params))):
                a2,a4,fabric = track.fabric(p,self.npoints,self.params[i],\
                                            self.model[i],solver=self.solver[i])

                #set nans to zero
                #a2 = a2.at[np.isnan(a2)].set(0)
                eigvals = np.linalg.eigvals(a2)
                
                eigvals = np.sort(eigvals,axis=1)


                
                if vertical:
                    plot_data = (eigvals[:,2],p.d)
                else:
                    plot_data = (p.d,eigvals[:,2])    


                if location == locations[0]:
                    ax.plot(*plot_data,label=self.legend[i],\
                            color=self.colors[i+1],linewidth=2)
                else:
                    ax.plot(*plot_data,\
                            color=self.colors[i+1],linewidth=2)


            if vertical:
                ax.set_xlabel('Largest Eigenvalue')
                ax.set_ylabel('Depth (m)')
                ax.invert_yaxis()
            else:
                ax.set_xlabel('Depth (m)')
                ax.set_ylabel('Largest Eigenvalue')
            ax.set_title(location + self.title)
            ax.grid(True,which='both',axis='both',linestyle='--',linewidth=0.5)
        fig.legend(loc='lower center',bbox_to_anchor=(0.5, -0.5),ncol=2)
        return fig


    def stream(self,ages=None):
        with open('./data/path2dOGdt10.pkl', 'rb') as f:
            path2d = pickle.load(f)


        depthsupper = np.array([5,25,50,75,100,150,200,250])
        depthslower = np.arange(375,1500,150)
        depths = np.concatenate((depthsupper,depthslower))
        # depths = np.array([400,1000])

        #depths = np.linspace(100,1800,16)
        times=-agedepth.depth2time(depths)


        
        paths=[]
        for t in times:
            nt = path2d['t'].size - np.abs(path2d['t'] - t).argmin()
            paths.append(track.stream(path2d,nt))


        for p in tqdm(paths):
            p.optimizeacc()
            p.Temperature(Ts=-30,Tb=-30)

            




        


        
        stoll_d,e_s,e_z,e_n = stoll.eigenvalues(dmin=depths[0],dmax=depths[-1])
        
        final_fabrics = []
        
        fig, ax = plt.subplots(3,1,figsize=(8,5))
        ax[0].scatter(stoll_d,e_n,s=5, label = 'EGRIP Data',**self.scatter_kwargs)
        ax[1].scatter(stoll_d,e_z,s=5,**self.scatter_kwargs)
        ax[2].scatter(stoll_d,e_s,s=5,**self.scatter_kwargs)
        woodcock = np.log(e_n/e_z)/np.log(e_z/e_s)

        # ax[1].scatter(stoll_d,woodcock,s=5,marker='.',color='black',alpha=0.5)

        dsave = 1048
        isave = np.abs(depths-dsave).argmin()

        ev_s = np.zeros((len(self.params),len(paths)))
        ev_n = np.zeros((len(self.params),len(paths)))
        ev_z = np.zeros((len(self.params),len(paths)))

        woodcock = np.zeros((len(self.params),len(paths)))

        a2save = np.zeros((len(self.params),len(paths),3,3))
        a4save = np.zeros((len(self.params),len(paths),3,3,3,3))


        for j in range(len(self.params)):

            depths = np.zeros(len(paths))
        
            for i in tqdm(range(len(paths))):

                if ages is None:
                    tind = 0
                else:
                    tind = np.abs(paths[i].t + ages[j]).argmin()
                
                a2,a4,fabric = track.fabric(paths[i],self.npoints,\
                                            self.params[j],\
                                                self.model[j],
                                                solver=self.solver[j],
                                                tind=tind,
                                                fabric0=self.fabric0)

                a2save[j,i,:,:] = a2[-1,:,:]
                a4save[j,i,:,:,:,:] = a4[-1,:,:,:,:]
                

                if i==isave:
                    final_fabrics.append(fabric)

                depths[i] = paths[i].d[-1]

                # update to handle nans in a2
                if np.isnan(a2).any():
                    ev_n[j,i] = np.nan
                    ev_s[j,i] = np.nan
                    ev_z[j,i] = np.nan
                else:
                    w,v = np.linalg.eig(a2[-1,:2,:2])
                    ev_n[j,i] = np.max(w)
                    ev_s[j,i] = np.min(w)
                    ev_z[j,i] = a2[-1,2,2]

                true_eigvals = np.abs(np.linalg.eigvalsh(a2[-1,:,:]))
                # sort descending
                true_eigvals = np.sort(true_eigvals)[::-1]

                woodcock[j,i] = np.log(true_eigvals[0]/true_eigvals[1])/np.log(true_eigvals[1]/true_eigvals[2])


            #ax.plot(depths,ev_z,color=colors[j+1],linestyle='--')
            ax[0].plot(depths,ev_n[j,:],color=self.colors[j+1],\
                       label=self.legend[j],linewidth=2)
            #ax.plot(depths,ev_s,color=colors[j+1],linestyle='-.')


            #woodcock = np.log(ev_n[j,:]/ev_z[j,:])/np.log(ev_z[j,:]/ev_s[j,:])
            ax[1].plot(depths,ev_z[j,:],color=self.colors[j+1],linewidth=2)
            ax[2].plot(depths,ev_s[j,:],color=self.colors[j+1],linewidth=2)



        for a in ax:
            a.grid(True,which='both',axis='both',linestyle='--',linewidth=0.5)
        ax[2].set_xlabel('Depth (m)')
        ax[0].set_ylabel(r'$e_{1}$')
        ax[1].set_ylabel(r'$e_{2}$')
        ax[2].set_ylabel(r'$e_{3}$')

        # remove xlabels from top 2 subplots
        ax[0].set_xticklabels([])
        ax[1].set_xticklabels([])

        # fig title
        fig.suptitle('EGRIP Eigenvalues $e_i$ vs. Depth' + self.title)

        # put legent outside both axes, at bottom with multiple columns
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        self.depths = depths
        self.stoll_d = stoll_d
        self.e_s = e_s
        self.e_z = e_z
        self.e_n = e_n
        self.ev_s = ev_s
        self.ev_n = ev_n
        self.ev_z = ev_z
        self.dsave = dsave

        return fig,final_fabrics


    def stream_violin(self):
        
        dmin = 550
        dind = np.abs(self.depths-dmin).argmin()
        dind_stoll = np.abs(self.stoll_d-dmin).argmin()
        fig, ax = plt.subplots(1,1,figsize=(4,4))

        ev_nl = []
        ev_sl = []
        ev_zl = []

        ev_nl.append(self.e_n[dind_stoll:])
        ev_sl.append(self.e_s[dind_stoll:])
        ev_zl.append(self.e_z[dind_stoll:])

        for i in range(len(self.params)):
            ev_nl.append(self.ev_n[i,:][dind:])

            ev_sl.append(self.ev_s[i,:][dind:])
            ev_zl.append(self.ev_z[i,:][dind:])

        # ax[0].violinplot(ev_nl,colors=colors)
        # ax[1].violinplot(woodcock)#,labels=['EGRIP']+legend)

        sns.violinplot(data=ev_nl,ax=ax,palette=self.colors,scale='width')
        sns.violinplot(data=ev_sl,ax=ax,palette=self.colors,scale='width')

        for a in ax:
            a.set_xticklabels(['EGRIP'] + self.legend, rotation=45, ha='right', rotation_mode='anchor')

            a.grid(True,which='both',axis='both',linestyle='--',linewidth=0.5)



        ax[0].set_ylabel('Largest Eigenvalue')
        ax[1].set_ylabel('Woodcock parameter')

        fig.suptitle('EGRIP '+str(dmin)+' to ' +str(int(self.depths[-1]))+' m' + self.title)

        return fig



    
    def plot_figures(self,final_fabrics):
        L=6
        mmax=8
        fig,axs = plt.subplots(1,len(self.params)+1,figsize=((len(self.params)+1)*2,3),\
                subplot_kw={'projection':ccrs.AzimuthalEquidistant(90,90)})

        for j in range(len(self.params)):

            

            n,m = final_fabrics[j]
                
            odf = BuildHarmonics(n,m,L,mmax)
            pcol = odf.plot(fig,axs[j+1],odf)
            axs[j+1].set_title(self.legend[j],fontsize=10)

        # plot EGRIP data
        import egrippolefigs
        n,m,w,depth = egrippolefigs.loadEGRIP(1)
        n = egrippolefigs.angle_correction(n)
        odf = BuildHarmonics(n,m,L,mmax)
        pcol = odf.plot(fig,axs[0],odf)
        axs[0].set_title('EGRIP Ice core',fontsize=10)

        fig.suptitle('Pole figures at ' + str(self.dsave) + 'm depth' + self.title)
    
        return fig


    def patterns(self,ngradu=3,dt=0.1,tmax=5):

        gradu = []

        gradu.append(np.array([[0.0,0.0,1.0],\
                        [0.0,0.0,0.0],\
                            [0.0,0.0,0.0]]))
        
        gradu.append(np.array([[0.5,0.0,0.0],\
                               [0.0,0.5,0.0],\
                                [0.0,0.0,-1.0]]))
        
        gradu.append(np.array([[1.0,0.0,0.0],\
                                 [0.0,0.0,0.0],\
                                  [0.0,0.0,-1.0]]))
        
        gradu = gradu[:ngradu]
        
        fig,axs = plt.subplots(len(gradu),len(self.params),\
                               figsize=(len(self.params)*2,3),\
                    subplot_kw={'projection':ccrs.AzimuthalEquidistant(90,90)})
        

        
        t,nt = mc_fab.time(dt,5)
        T = -30*np.ones(nt)
        dt_tile = np.tile(dt,(nt,))
        for i in range(len(gradu)):
            gradu_tile = np.tile(gradu[i],(nt,1,1))
            for j in range(len(self.params)):
            
                x = self.params[j]
                if isinstance(x,str):
                    if x == 'Richards':
                        x_tile = mc_fab.parameters.Richards2021(T)
                    elif x == 'Elmer':
                        x_tile = mc_fab.parameters.Elmer(T)
                else:    
                    x_tile = np.tile(x,(nt,1))

                

                



                n,m,a2,a4 = mc_fab.static_mc.solve(self.npoints,gradu_tile,\
                                                dt_tile,x_tile,\
                                                    self.model[j])
                

                
                if len(gradu) == 1:
                    ax = axs[j]
                else:
                    ax = axs[i,j]

                odf = BuildHarmonics(n[-1],m[-1],6,8)
                pcol = odf.plot(fig,ax,odf)
        
        
        for j in range(len(self.params)):
            if len(gradu) == 1:
                ax = axs[j]
            else:
                ax = axs[0,j]
            ax.set_title(self.legend[j])


        deformation =['Simple shear','Unconfined compression','Pure shear']
        
        #for i in range(len(gradu)):
            #axs[i,0].text(0,-7000e3,deformation[i])


        fig.suptitle(r'Simple shear: pole figures at $\gamma = ' + str(tmax) + '$' + self.title)
        return fig












    


