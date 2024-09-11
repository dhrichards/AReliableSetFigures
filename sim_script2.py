#%%
import numpy as np# numpy for arrays
from matplotlib import pyplot as plt
from simulations import simulation
import seaborn as sns
import mcfab as mc
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 12,
    "figure.autolayout" : False,
    
})


npoints = 2000



# colors = ['#1f77b4','#03045e', '#0077b6', '#00b4d8',\
#                            '#90e0ef','#caf0f8','#f72585','#7209b7',\
#                            '#3a0ca3','#4361ee','#4cc9f0']


def legfmt(x):
    return '$\\tilde{\\lambda}=' + str(x[2]) + '$, $E_{ca} = ' +str(int(x[-1]))+'$'

def legfmt2(x):
    return '$\\tilde{\\lambda}=' + str(x[2]) + '$, $\\tilde{\\Gamma} =' \
          + str(x[2]) + '$'


def legfmt3(x,model):
    #convert to 3sig fig
    format = '%.2g'
    iotaD = format % x[0]
    alphaS = x[1]/mc.parameters.iota(x[4],x[5])
    alphaS = format % alphaS
    lamb = format % x[2]
    return '$\iota_D=' + iotaD + '$, $\\alpha_S= ' +str(alphaS)+'$'\
            + ', $\\tilde{\\lambda} = ' + lamb + '$'

params = []
age = None
if age:
    params.append(mc.GrainClass(alpha=0.0,lamb=0.00,Eca=1e2,power=3))
    params.append(mc.MacroscopicClass(alphaD=2.0,alphaS=0,lamb=0.2,Eca=1,power=3))
    params.append(mc.MacroscopicClass(alphaD=0.0,alphaS=1.1,lamb=0.36,Eca=1e2,power=3))
    params.append(mc.GrainClass(alpha=1.0,lamb=0.25,Eca=1e2,power=3))
    params.append(mc.MacroscopicClass(alphaD=0.0,alphaS=1.0,lamb=0.3,Eca=1e2,power=3))
    params.append(mc.MacroscopicClass(alphaD=0.0,alphaS=1.5,lamb=0.45,Eca=1e2,power=3))
    params.append(mc.MacroscopicClass(alphaD=0.0,alphaS=1.0,lamb=0.25,Eca=1e2,power=3))
    
else:
    params.append(mc.GrainClass(alpha=0.0,lamb=0.03,Eca=1e2,power=3))
    params.append(mc.MacroscopicClass(alphaD=2.6,alphaS=0,lamb=0.32,Eca=1,power=3))
    params.append(mc.MacroscopicClass(alphaD=0.0,alphaS=1.1,lamb=0.36,Eca=1e2,power=3))
    params.append(mc.GrainClass(alpha=1.0,lamb=0.32,Eca=1e2,power=3))
    params.append(mc.MacroscopicClass(alphaD=0.0,alphaS=1.15,lamb=0.45,Eca=1e2,power=3))
    params.append(mc.MacroscopicClass(alphaD=0.0,alphaS=1.47,lamb=0.49,Eca=1e2,power=3))
    params.append(mc.MacroscopicClass(alphaD=0.0,alphaS=1.1,lamb=0.36,Eca=1e2,power=3))


ages = [age]*len(params)

# 0 = sachs, 1= golf, 2 = caffe, 3 = glen, 4 = RathmannFull, 5 = Petit, 6= RathmannMartin 7= Rathmann2 8 = Golf2
model = [0,3,2,0,1,4,5]
# model = [3,2,0,1,5]
# model = [2,2]
solver = ['mc']*len(params)
# solver[3] = 'mcnp'
solver[4] = 'mcnp'



# legend = ['Estar/Glen','CAFFE','Sachs','GOLF','Rathmann']
legend = ['Taylor', 'Estar/Glen','CAFFE','Sachs','GOLF','Rathmann \n Unapprox.','Rathmann\n Petit']
# legend = ['Original','Modified']


colors = sns.color_palette("deep",len(params)+1)

if age:
    fileprefix = 'young'
else:
    fileprefix = ''
sim = simulation(npoints, params, legend, model,colors=colors,solver=solver)

fig = sim.divide(1)
fig.savefig('./images/'+ fileprefix +'divides.png', bbox_inches='tight',dpi=400)


if age:
    fig,fabrics = sim.stream(ages=ages)
else:
    fig,fabrics = sim.stream()


fig.savefig('./images/' + fileprefix +'stream.pdf', bbox_inches='tight')
fig = sim.plot_figures(fabrics)
fig.savefig('./images/' + fileprefix +'fabrics.png', bbox_inches='tight',dpi=400)


#%%
# Categorical scatter plot
colors = sns.color_palette("deep",3)
dmin = 550
dind = np.abs(sim.depths-dmin).argmin()
dind_stoll = np.abs(sim.stoll_d-dmin).argmin()

fig,ax = plt.subplots(1,1,figsize=(7,4),sharex=True)

scatter_kwargs = {'marker': 'x', \
                                 's': 50,\
}



meanevs = np.zeros((len(sim.params),3))
for i in range(len(sim.params)):
    meanevs[i,0] = np.mean(sim.ev_n[i,:][dind:])
    meanevs[i,1] = np.mean(sim.ev_s[i,:][dind:])
    meanevs[i,2] = np.mean(sim.ev_z[i,:][dind:])

std_n = np.std(sim.e_n[dind_stoll:])
std_s = np.std(sim.e_s[dind_stoll:])
std_z = np.std(sim.e_z[dind_stoll:])

ax.scatter(legend,meanevs[:,0],label='Largest',color=colors[0],**scatter_kwargs)
ax.scatter(legend,meanevs[:,1],label='Middle',color=colors[1],**scatter_kwargs)
ax.scatter(legend,meanevs[:,2],label='Smallest',color=colors[2],**scatter_kwargs)

# add horizontal line at EGRIP value including std dev



ax.axhline(y=np.mean(sim.e_n[dind_stoll:]),color=colors[0],linestyle='--')
ax.axhline(y=np.mean(sim.e_s[dind_stoll:]),color=colors[1],linestyle='--')
ax.axhline(y=np.mean(sim.e_z[dind_stoll:]),color=colors[2],linestyle='--')

ax.axhspan(np.mean(sim.e_n[dind_stoll:])-std_n,np.mean(sim.e_n[dind_stoll:])+std_n,color=colors[0],alpha=0.2)
ax.axhspan(np.mean(sim.e_s[dind_stoll:])-std_s,np.mean(sim.e_s[dind_stoll:])+std_s,color=colors[1],alpha=0.2)
ax.axhspan(np.mean(sim.e_z[dind_stoll:])-std_z,np.mean(sim.e_z[dind_stoll:])+std_z,color=colors[2],alpha=0.2)


ax.set_ylabel('Eigenvalues')
ax.set_ylim([-0.05,0.9])



# y log
# ax.set_yscale('log')
# ax.set_ylim([1e-2,1e0])

# broken y axis
# axs[0].set_ylim([np.min(meanevs[:,2])-0.02,np.max(meanevs[:,0])+0.08])
# axs[1].set_ylim([0,np.max(meanevs[:,1])+0.02])
# axs[0].spines['bottom'].set_visible(False)
# axs[1].spines['top'].set_visible(False)
# axs[0].xaxis.tick_top()

# # cut-out slanted lines
# d = .015  # how big to make the diagonal lines in axes coordinates
# kwargs = dict(transform=axs[0].transAxes, color='k', clip_on=False)
# axs[0].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# axs[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

# kwargs.update(transform=axs[1].transAxes)  # switch to the bottom axes
# axs[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# axs[1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# #ylabel spread over both
# fig.text(0.04, 0.5, 'Eigenvalues', va='center', rotation='vertical')

title = 'Mean eigenvalues at EGRIP, '+str(dmin)+' to ' +str(int(sim.depths[-1]))+' m'
if age:
    title += ', for ' + str(ages[0]) + ' year old ice stream'
fig.suptitle(title)


#Add custom legend to include EGRIP horizontal line
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
patch = Patch(color='k',alpha=0.2)
line = Line2D([0], [0], color='k', lw=2,linestyle='--',)
marker = Line2D([0], [0], color='k', lw=2,linestyle='none',marker='x',markersize=7)
ax.legend([(patch,line),marker],['EGRIP','Models'])

if age:
    fig.savefig('./images/streammeans' + str(ages[0]) + '.pdf', bbox_inches='tight')
    fig.savefig('./images/streammeans' + str(ages[0]) + '.png', bbox_inches='tight',dpi=400)
else:
    fig.savefig('./images/streammeans.pdf', bbox_inches='tight')
    fig.savefig('./images/streammeans.png', bbox_inches='tight',dpi=400)

#%%

import cartopy.crs as ccrs
L=8
mmax=8
nrows = 2
ncols = 4

ngrains = 800
fig = plt.figure(figsize=(9,5))

gs0 = fig.add_gridspec(1,2 ,width_ratios=[1,2.8],wspace=0.1)
gs1 = gs0[1].subgridspec(nrows,ncols,hspace=0.1,wspace=0.1)

axs=[]
ax = fig.add_subplot(gs0[0],projection=ccrs.AzimuthalEquidistant(90,90))
axs.append(ax)
for i in range(nrows):
    for j in range(ncols):
        ax = fig.add_subplot(gs1[i,j],projection=ccrs.AzimuthalEquidistant(90,90))
        axs.append(ax)


axs[-1].remove()

# axs = [   fig.add_subplot(gs0[0],projection=ccrs.AzimuthalEquidistant(90,90)),\
#         fig.add_subplot(gs1[0,0],projection=ccrs.AzimuthalEquidistant(90,90)),\
#         fig.add_subplot(gs1[0,1],projection=ccrs.AzimuthalEquidistant(90,90)),\
#         fig.add_subplot(gs1[0,2],projection=ccrs.AzimuthalEquidistant(90,90)),\
#         fig.add_subplot(gs1[1,0],projection=ccrs.AzimuthalEquidistant(90,90)),\
#         fig.add_subplot(gs1[1,1],projection=ccrs.AzimuthalEquidistant(90,90)),\
#         fig.add_subplot(gs1[1,2],projection=ccrs.AzimuthalEquidistant(90,90)),\
#             fig.add_subplot(gs1[0,3],projection=ccrs.AzimuthalEquidistant(90,90))]
# # axs = [fig.add_subplot(gs0[0]),\

# fig,axs = plt.subplots(2,ncol,figsize=(9,5),\
#         subplot_kw={'projection':ccrs.AzimuthalEquidistant(90,90)})

import seaborn as sns
# colors = sns.color_palette("cmo.ice",len(params)+1)
colors = sns.color_palette("deep",len(params)+1)

# dark blue
color = 'k'
scatter_kwargs = {'marker': 'o', \
                                 'alpha': 0.3,\
                                    'edgecolors':'none',
                                    's':10}



# shuffle = [0,5,6,7,1,2,3,4]
for j in range(len(sim.params)):
    

    # ax = axs[(j+1)//ncol,(j+1)%ncol]
    ax = axs[j+1]

    n,m = fabrics[j]
    lon = np.degrees(np.arctan2(n[:,1],n[:,0]))
    lat = np.degrees(np.abs(np.arcsin(n[:,2])))

    # odf = mc.BuildHarmonics(n,m,L,mmax)
    # pcol = odf.plot(fig,ax,True,colorbar=False,alpha=0.7,cmap='Greys')
    
    # odf = mc.BuildHarmonics(n,m,L,mmax)
    # pcol = odf.plot(fig,ax,odf,colorbar=False)
    ax.scatter(lon[:ngrains],lat[:ngrains],transform=ccrs.PlateCarree(),color=color,**scatter_kwargs)
    
    # remove white background from title
    ax.set_title(sim.legend[j],backgroundcolor='none')

    ax.set_aspect('equal')
    ax.axis('off')
    kwargs_gridlines = {'ylocs':np.arange(-89,90+30,30), \
                        'xlocs':np.arange(-360,+360,45),\
                            'linewidth':0.5, 'color':'black', 'alpha':0.25, \
                                'linestyle':'-'}
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(),**kwargs_gridlines)#,xlocs=[s_dir,s_dir+90,s_dir+180,s_dir+270])
    # ax.set_extent([-180,180,0,0], crs=ccrs.PlateCarree())

    gl.ylim = (0,90)
    ax.set_extent([-180,180,0,90], crs=ccrs.PlateCarree())
    
    # _,_,fgrid = odf.grid()
    # fmax = np.max(fgrid)
    # fmaxtext = '%.1f' % fmax
    # ax.text(0, -12e6,r'$f_{max} = ' + fmaxtext + '$',fontsize=10,ha='center')


# plot EGRIP data
import egrippolefigs
n,m,w,depth = egrippolefigs.loadEGRIP(1)
n = egrippolefigs.angle_correction(n)

lon = np.degrees(np.arctan2(n[:,1],n[:,0]))
lat = np.degrees(np.arcsin(n[:,2]))



axs[0].scatter(lon[:ngrains],lat[:ngrains],transform=ccrs.PlateCarree(),color=color,**scatter_kwargs)
axs[0].set_aspect('equal')
axs[0].axis('off')
axs[0].set_extent([-180,180,0,90], crs=ccrs.PlateCarree())
kwargs_gridlines = {'ylocs':np.arange(-89,90+30,30), \
                    'xlocs':np.arange(-360,+360,45),\
                        'linewidth':0.5, 'color':'black', 'alpha':0.25, \
                            'linestyle':'-'}


gl = axs[0].gridlines(crs=ccrs.PlateCarree(),**kwargs_gridlines)#,xlocs=[s_dir,s_dir+90,s_dir+180,s_dir+270])
# ax.set_extent([-180,180,0,0], crs=ccrs.PlateCarree())

gl.ylim = (0,90)
# Egrip ice core at 1320 m
axs[0].set_title('EGRIP Ice core \n depth = ' + str(int(depth)) + ' m') 
# _,_,fgrid = odf.grid()
# fmax = np.max(fgrid)
# fmaxtext = '%.1f' % fmax
# axs[0,0].text(0, -12e6,r'$f_{max} = ' + fmaxtext + '$',fontsize=10,ha='center')

# add colorbar
# cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
# fig.colorbar(pcol, cax=cax, orientation='vertical',label='ODF')
# # add custom labels from 0 to f_max
# cax.set_yticks([0,fmax])
# cax.set_yticklabels(['0',r'$f_{max}$'])

#add east and north labels
axs[0].text(180, 0,r'E',ha='center',transform=ccrs.PlateCarree(),va='center')
axs[0].text(90,0,r'S',ha='center',transform=ccrs.PlateCarree(),va='center')


# fig.suptitle('Pole figures at ' + str(sim.dsave) + 'm depth' + sim.title)



fig.savefig('./images/polefigs2.pdf', bbox_inches='tight')
fig.savefig('./images/polefigs2.png', bbox_inches='tight',dpi=400)


#%% Special pole fig fig for many params
import cartopy.crs as ccrs

L=6
mmax=6
nrows = 2
ncols = 4
cmap = 'viridis'

vmax = 0.4
npf =2 # number of EGRIP pole figures to plot

fig = plt.figure(figsize=(10,5))

gs0 = fig.add_gridspec(1,2 ,width_ratios=[npf,ncols],wspace=0.1)
gs1 = gs0[1].subgridspec(nrows,ncols,hspace=0.1,wspace=0.1)
gs2 = gs0[0].subgridspec(1,npf,hspace=0.1,wspace=0.1)

axs=[]

proj = ccrs.AzimuthalEquidistant(90,90); hemi = True
# proj = ccrs.Orthographic(0,-45); hemi = False
for i in range(npf):
    ax = fig.add_subplot(gs2[i],projection=proj)
    axs.append(ax)

for i in range(nrows):
    for j in range(ncols):
        ax = fig.add_subplot(gs1[i,j],projection=proj)
        axs.append(ax)


axs[-1].remove()

# shuffle = [4,5,6,3,0,1,2]
for j in range(len(sim.params)):
    

    ax = axs[j+npf]

    n,m = fabrics[j]
        
    odf = mc.BuildHarmonicsSpecFab(n,m,L,mmax)
    pcol = odf.plot(fig,ax,hemi,colorbar=False,cmap=cmap,vmax=vmax)
    ax.set_title(sim.legend[j])
    _,_,fgrid = odf.grid()
    fmax = np.max(fgrid)
    fmaxtext = '%.1f' % fmax
    # ax.text(0, -12e6,r'$f_{max} = ' + fmaxtext + '$',fontsize=10,ha='center')

# plot EGRIP data
import egrippolefigs

n2,m,w,depth2 = egrippolefigs.loadEGRIP(2)
n1,m,w,depth1 = egrippolefigs.loadEGRIP(1)

depths = [depth1,depth2]
ns = [n1,n2]

for i in range(npf):
    ax = axs[i]
    n = ns[i]
    n = egrippolefigs.angle_correction(n)
    n = np.vstack((n,-n))

    odf = mc.BuildHarmonicsSpecFab(n,np.ones(n.shape[0]),L,mmax)
    pcol = odf.plot(fig,ax,hemi,colorbar=False,cmap=cmap,vmax=vmax)
    ax.set_title('EGRIP core\n at ' + str(int(depths[i])) + 'm')

    _,_,fgrid = odf.grid()
    fmax = np.max(fgrid)
    fmaxtext = '%.1f' % fmax
    # ax.text(0, -12e6,r'$f_{max} = ' + fmaxtext + '$',fontsize=10,ha='center')


# add colorbar
cax = fig.add_axes([0.8, 0.2, 0.01, 0.2])
fig.colorbar(pcol, cax=cax, orientation='vertical',label='ODF')
# add custom labels from 0 to f_max
cax.set_yticks([0,vmax])
# cax.set_yticklabels(['0',r'$f_{max}$'])

# add little compass bottom left

# white text
# fig.suptitle('Pole figures at ' + str(sim.dsave) + 'm depth' + sim.title)

fig.savefig('./images/' + fileprefix + 'polefigs.pdf', bbox_inches='tight')
fig.savefig('./images/' + fileprefix + 'polefigs.png', bbox_inches='tight',dpi=400)

#%%

import track
import divide_data
ncols = 2
split = 3
fig,axs = plt.subplots(ncols=ncols,figsize=(8,4))
location = 'GRIP'
dt = 100
vertical = True
data = divide_data.data(location)

if ncols == 1:
    axs = [axs]

colors = sns.color_palette("deep",len(params)+1)
for ax in axs:
    if vertical:
        plot_data = (data.largest_ev, data.largest_ev_d)
    else:
        plot_data = (data.largest_ev_d, data.largest_ev)

    ax.scatter(*plot_data,label='Ice cores',s=30,color=colors[0],marker='x')

p = track.divide(dt=dt,location=location,dh=0.9)
p.depth()
for i in range(len(sim.params)):

    if ncols > 1:
        if i<split:
            ax = axs[0]
            j = i
        else:
            ax = axs[1]
            j = i-3
    else: 
        ax = axs[0]
        j = i



    a2,a4,fabric = track.fabric(p,npoints,params[i],model[i],solver=solver[i])


    #set nans to zero
    #a2 = a2.at[np.isnan(a2)].set(0)
    eigvals = np.linalg.eigvals(a2)
    
    eigvals = np.sort(eigvals,axis=1)


    
    if vertical:
        plot_data = (eigvals[:,2],p.d)
    else:
        plot_data = (p.d,eigvals[:,2])    


    
    ax.plot(*plot_data,label=legend[i],\
            color=colors[i+1],linewidth=2)

for ax in axs:
    if vertical:
        ax.set_xlabel('Largest Eigenvalue')
        ax.set_ylim(-100,2500)
        ax.invert_yaxis()
        ax.grid(True,which='both',axis='both',linestyle='--',linewidth=0.5)
    else:
        ax.set_xlabel('Depth (m)')
        ax.set_ylim(0.3,1.01)
        ax.grid(True,which='both',axis='both',linestyle='--',linewidth=0.5)

    ax.legend()#loc='lower center',bbox_to_anchor=(0.5, -0.3),fontsize=10,ncol=2)
    


fig.suptitle('GRIP ice divide')
if vertical:
    axs[0].set_ylabel('Depth (m)')
else:
    axs[0].set_ylabel('Largest Eigenvalue')

fig.savefig('./images/' + fileprefix + 'dividedouble.png', bbox_inches='tight',dpi=400)
fig.savefig('./images/' + fileprefix + 'dividedouble.pdf', bbox_inches='tight')


#%%
import pandas as pd

#vertical stream eigenvalue plot
colors = sns.color_palette("deep",len(params)+1)
scatter_kwargs = {'marker': 'o', \
                               'alpha': 0.3, 'color': colors[0],\
                                'edgecolors':'none'}

# make two subplots, first one 60% width

fig, ax = plt.subplots(2,3,figsize=(9,6))
for i in range(2):
    if i == 0:
        ax[i,0].scatter(sim.e_s,sim.stoll_d,s=5, label = 'EGRIP Data',**scatter_kwargs)
    else:
        ax[i,0].scatter(sim.e_s,sim.stoll_d,s=5,**scatter_kwargs)
    ax[i,1].scatter(sim.e_z,sim.stoll_d,s=5,**scatter_kwargs)
    ax[i,2].scatter(sim.e_n,sim.stoll_d,s=5,**scatter_kwargs)


# Add in special points for EGRIP pole figures
import egrippolefigs

n2,m,a2_2,depth2 = egrippolefigs.loadEGRIP(2)
n1,m,a2_1,depth1 = egrippolefigs.loadEGRIP(1)

deptspf = [depth1,depth2]
a2pf = [a2_1,a2_2]
spf = 40
for i in range(2):
    for j in range(3):
        if i == 0 and j == 0:
            ax[i,j].scatter(a2pf[i][2-j],deptspf[i],s=spf,marker='x',color='k',label="EGRIP Pole \n figure")
        else:
            ax[i,j].scatter(a2pf[0][2-j],deptspf[0],s=spf,marker='x',color='k')
        ax[i,j].scatter(a2pf[1][2-j],deptspf[1],s=spf,marker='x',color='k')
# # plot rolling average and rolling std dev
# window = 30
# for data,i in zip([sim.e_s,sim.e_z,sim.e_n],range(3)):
#     ts = pd.Series(data,index=sim.stoll_d)
#     rolling = ts.rolling(window=window)
#     rolling_mean = rolling.mean()
#     rolling_std = rolling.std()
#     if i == 0:
#         ax[i].plot(rolling_mean,sim.stoll_d,linewidth=2,label='EGRIP data',color=colors[0])
#     else:
#         ax[i].plot(rolling_mean,sim.stoll_d,linewidth=2,color=colors[0])
#     ax[i].fill_betweenx(sim.stoll_d,rolling_mean-rolling_std,rolling_mean+rolling_std,alpha=0.2,color=colors[0])


jtoplot = [0,1,2]
jtoplot = [3,4,5,6]
for j in range(len(sim.params)):
# for j in jtoplot:
    if j ==10:
        special = {'linestyle':'-.'}
    else:
        special = {}

    if j < 3:
        i = 0
    else:
        i = 1


    

    ax[i,0].plot(sim.ev_s[j,:],sim.depths,color=colors[j+1],label=legend[j],linewidth=2,**special)
    ax[i,1].plot(sim.ev_z[j,:],sim.depths,color=colors[j+1],linewidth=2,**special)
    ax[i,2].plot(sim.ev_n[j,:],sim.depths,color=colors[j+1],linewidth=2,**special)

for a in ax.flatten():
    a.grid(True,which='both',axis='both',linestyle='--',linewidth=0.5)
    a.set_ylim([-50,1550])

for i in range(2):

#flip y axis
    ax[i,0].invert_yaxis()
    ax[i,1].invert_yaxis()
    ax[i,2].invert_yaxis()

# show y labels only on ax[0]
    ax[i,1].set_yticklabels([])
    ax[i,2].set_yticklabels([])

    ax[i,0].set_ylabel('Depth (m)')



ax[1,0].set_xlabel(r'Smallest eigenvalue')
ax[1,1].set_xlabel(r'Middle eigenvalue')
ax[1,2].set_xlabel(r'Largest eigenvalue')

# fig title
fig.suptitle('EGRIP Eigenvalues $e_i$ vs. Depth')

# put legent outside axes, to the right
fig.legend(loc='center right', bbox_to_anchor=(1.1, 0.5), ncol=1)
# fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)


fig.savefig('./images/' + fileprefix + 'streamvertical.pdf', bbox_inches='tight')
fig.savefig('./images/' + fileprefix + 'streamvertical.png', bbox_inches='tight',dpi=400)


#%%
# Divide error
import track
import divide_data
location = 'GRIP'
dt = 100
data = divide_data.data(location)


        # plot_data = (data.largest_ev, data.largest_ev_d)
   
p = track.divide(dt=dt,location=location,dh=0.9)
p.depth()

error = np.zeros((len(sim.params),len(data.largest_ev_d)))
for i in range(len(sim.params)):

    a2,a4,fabric = track.fabric(p,npoints,params[i],model[i],solver=solver[i])

    eigvals = np.linalg.eigvals(a2)
    eigvals = np.sort(eigvals,axis=1)

    # find nearest depth for each d
    for j in range(len(data.largest_ev_d)):
        d = data.largest_ev_d[j]
        ind = np.abs(p.d-d).argmin()
        error[i,j] = np.abs(eigvals[ind,2]-data.largest_ev[j])



error_sum = np.sum(error,axis=1)
#normalise relative to Estar/Glen
error_sum = error_sum/error_sum[1]
fig,ax = plt.subplots(1,1,figsize=(7.5,4))
colors = sns.color_palette("deep",len(params))

# bar chart of error
ax.bar(legend,error_sum*100,color=colors)
ax.set_ylabel('Normalised error (\%)')
fig.suptitle('Normalised error in largest eigenvalue at GRIP divide')

# grid
ax.grid(True,which='both',axis='y',linestyle='--',linewidth=0.5)


#Error stream

dmin = 550
dind = np.abs(sim.depths-dmin).argmin()
dind_stoll = np.abs(sim.stoll_d-dmin).argmin()

dlen = len(sim.depths) - dind



error_stream = np.zeros((len(sim.params),dlen))
for i in range(len(sim.params)):

    for j in range(dlen):
        dind_stoll_j = np.abs(sim.stoll_d-sim.depths[j+dind]).argmin()
        error_stream[i,j] = np.abs(sim.ev_n[i,j+dind]-sim.e_n[dind_stoll_j]) + \
                            np.abs(sim.ev_s[i,j+dind]-sim.e_s[dind_stoll_j]) + \
                            np.abs(sim.ev_z[i,j+dind]-sim.e_z[dind_stoll_j]) 
        

fig,ax = plt.subplots(1,1,figsize=(7.5,4))

colors = sns.color_palette("deep",len(params))

error_stream_sum = np.sum(error_stream,axis=1)
#normalise relative to Estar/Glen
error_stream_sum = error_stream_sum/error_stream_sum[1]

ax.bar(legend,error_stream_sum*100,color=colors)
ax.set_ylabel('Normalised Error (\%)')
fig.suptitle('Normalised Error in eigenvalues at EGRIP, '+str(dmin)+' to ' +str(int(sim.depths[-1]))+' m')

# #add values to top of bars
# for i in range(len(sim.params)):
#     ax.text(i,error_stream_sum[i]*100+0.5,'%.2f' % (error_stream_sum[i]*100),ha='center')
#grid
ax.grid(True,which='both',axis='y',linestyle='--',linewidth=0.5)



# combined bar
import seaborn as sns
import cmocean.cm as cmo
fig,ax = plt.subplots(1,1,figsize=(7.5,4))

# use cmocean ice colormap


# 2 bars next to each other
width = 0.4
ax.bar(x=np.arange(len(params))-width/2,height=error_sum*100,width=width,label='GRIP',color=cmo.ice(0.9))
ax.bar(x=np.arange(len(params))+width/2,height=error_stream_sum*100,width=width,label='EGRIP, '+str(dmin)+' to '+str(int(sim.depths[-1]))+' m',color=cmo.ice(0.5))


ax.set_xticks(np.arange(len(params)))
ax.set_xticklabels(legend)

ax.set_ylabel('Normalised Error (\%)')

#legend
ax.legend()

#grid
ax.grid(True,which='both',axis='y',linestyle='--',linewidth=0.5)

fig.savefig('./images/' + fileprefix + 'errorcombined.pdf', bbox_inches='tight')
fig.savefig('./images/' + fileprefix + 'errorcombined.png', bbox_inches='tight',dpi=400)