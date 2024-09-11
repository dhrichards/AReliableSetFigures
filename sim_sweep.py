#%%
import numpy as np# numpy for arrays
from matplotlib import pyplot as plt
from simulations import simulation
import seaborn as sns
import mcfab as mc
import cmocean
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 10,
    "figure.autolayout" : False,
    
})
#%%
nsweep = 12
paramvals = [np.linspace(0,8,nsweep),
            np.concatenate((np.linspace(0,1,nsweep//2),np.linspace(1,10,nsweep//2)))]
label = ['$\\alpha_D$ \n Lattice rotation \n due to deformation',
         '$\\tilde{\\lambda}$\nRotational\nrecrystallization',
            '$\\tilde{\\beta}$\nMigration\nrecrystallization',
]
scale = ['linear','linear','linear']




alphaD0 = 2.6
lamb0 = 0.32
beta0 = 0.0


# x0 = mc.GrainClass(alpha=1.0,lamb=0.32,beta=0,Eca=1e2,power=3)
# x0 = mc.MacroscopicClass(alphaD=0.0,alphaS=1.1,lamb=0.36,Eca=1e2,power=1)
model = 0
# x0 = np.array(x0)
params = []

for i in range(len(paramvals)):
    param = []
    for j in range(len(paramvals[i])):
        albE = [alphaD0,lamb0,beta0]
        if i == 0: # Checking deformation so set alphaS = 0
            albE[1] = 0

        albE[i] = paramvals[i][j]
        x = mc.MacroscopicClass(alphaD=albE[0],alphaS=0,lamb=albE[1],beta=albE[2],Eca=1e2,power=3)
        
        param.append(x)

    params.append(param)



sims = []
npoints = 1000
for j in range(len(paramvals)):


    models = [model]*len(params[j])
    solver = ['mc']*len(params[j])


    colors = sns.color_palette("deep",len(params[j])+1)



    def legfmt(x):
        return '$\\tilde{\\lambda}=' + str(x[2]) + '$, $E_{ca} = ' +str(int(x[-1]))+'$'

    legend = [legfmt(x) for x in params[j]]

    
    sims.append(simulation(npoints, params[j], legend, models,colors=colors,solver=solver)
                )

    fig,fabrics = sims[j].stream()




#%%
dmin = 550

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
#     "font.size" : 10,
    
# })


#label per param index
fig, ax = plt.subplots(1,len(paramvals),figsize=(9,2))

for j in range(len(paramvals)):

    dind = np.abs(sims[j].depths-dmin).argmin()
    dind_stoll = np.abs(sims[j].stoll_d-dmin).argmin()

    ev_nl = []
    ev_sl = []
    ev_zl = []

    ev_nl.append(sims[j].e_n[dind_stoll:])
    ev_sl.append(sims[j].e_s[dind_stoll:])
    ev_zl.append(sims[j].e_z[dind_stoll:])

    for i in range(len(sims[j].params)):
        ev_nl.append(sims[j].ev_n[i,:][dind:])

        ev_sl.append(sims[j].ev_s[i,:][dind:])
        ev_zl.append(sims[j].ev_z[i,:][dind:])

    # ax[0].violinplot(ev_nl,colors=colors)
    # ax[1].violinplot(woodcock)#,labels=['EGRIP']+legend)

    ev_nl_mean = np.zeros(len(ev_nl))
    ev_sl_mean = np.zeros(len(ev_nl))
    ev_zl_mean = np.zeros(len(ev_nl))

    ev_nl_std = np.zeros(len(ev_nl))
    ev_sl_std = np.zeros(len(ev_nl))
    ev_zl_std = np.zeros(len(ev_nl))



    for i in range(len(ev_nl)):
        ev_nl_mean[i] = np.mean(ev_nl[i])
        ev_sl_mean[i] = np.mean(ev_sl[i])
        ev_zl_mean[i] = np.mean(ev_zl[i])

        ev_nl_std[i] = np.std(ev_nl[i])
        ev_sl_std[i] = np.std(ev_sl[i])
        ev_zl_std[i] = np.std(ev_zl[i])

    if j == 0:
        ax[j].plot(paramvals[j],ev_nl_mean[1:],color=colors[0],label='Largest Eigenvalue')
        ax[j].plot(paramvals[j],ev_sl_mean[1:],color=colors[1],label='Smallest Eigenvalue')
        ax[j].plot(paramvals[j],ev_zl_mean[1:],color=colors[2],label='Middle Eigenvalue')

        # ax[j].axhline(ev_nl_mean[0],linestyle='--',color=colors[0],linewidth=0.75, label='EGRIP data')
        # ax[j].axhline(ev_sl_mean[0],linestyle='--',color=colors[1],linewidth=0.75)
        # ax[j].axhline(ev_zl_mean[0],linestyle='--',color=colors[2],linewidth=0.75)
    else:
        ax[j].plot(paramvals[j],ev_nl_mean[1:],color=colors[0])
        ax[j].plot(paramvals[j],ev_sl_mean[1:],color=colors[1])
        ax[j].plot(paramvals[j],ev_zl_mean[1:],color=colors[2])

    ax[j].set_xlabel(label[j])

    ax[j].set_xscale(scale[j])
    
    ax[j].set_xscale(scale[j])

        # ax[j].axhline(ev_nl_mean[0],linestyle='--',color=colors[0],linewidth=0.75)
        # ax[j].axhline(ev_sl_mean[0],linestyle='--',color=colors[1],linewidth=0.75)
        # ax[j].axhline(ev_zl_mean[0],linestyle='--',color=colors[2],linewidth=0.75)



    ax[j].set_ylim(0.0,1.0)
    #add grid
    ax[j].grid(True,which='both',axis='both',linestyle='--',linewidth=0.5)
    if j>0:
        ax[j].set_yticklabels([])


ax[0].set_ylabel('Eigenvalue')
# ax[0].set_xlabel('$\\alpha$')
# ax[1].set_xlabel('$\\tilde{\\lambda}$')

# ax[2].set_xlabel('$\\beta$')
# ax[3].set_xlabel('$E_{cc}$')

# ax[4].set_xlabel('$E_{ca}$')

# ax[4].set_xscale('log')




# ax[0].axvline(1.0,linestyle='--',color='black',linewidth=1.5,label='Value used')
# ax[1].axvline(0.34,linestyle='--',color='black',linewidth=1.5)
# ax[2].axvline(0.0,linestyle='--',color='black',linewidth=1.5)
# ax[3].axvline(1.0,linestyle='--',color='black',linewidth=1.5)
# ax[4].axvline(100.0,linestyle='--',color='black',linewidth=1.5)

#legend on right outside
# fig.legend(loc='center right',bbox_to_anchor=(1.11, 0.5),fontsize=10,ncol=1)

#0 = sachs, 1= golf, 2 = caffe, 3 = glen, 4 = RathmannFull, 5 = Petit, 6= RathmannMartin 7= Rathmann2 8 = Golf2

# if model ==0 and alpha0>0.5 :
#     name = 'Sachs model'
#     filename = 'sachs'
# elif model ==0 and alpha0<0.5 :
#     name = 'Taylor model'
#     filename = 'taylor'
# elif model ==1:
#     name = 'GOLF flow law'
#     filename = 'golf'
# elif model ==2:
#     name = 'CAFFE flow law'
#     filename = 'caffe'
# elif model ==3:
#     name = 'Glens flow law'
#     filename = 'glen'
# elif model ==4:
#     name = 'Rathmann Unapprox. flow law'
#     filename = 'rathmannfull'
# elif model ==5:
#     name = 'Rathmann Petit flow law'
#     filename = 'petit'



fig.suptitle('Parameter sensitivity: Average eigenvalues at EGRIP from ' +str(dmin)+' to ' +str(int(sims[0].depths[-1]))+' m',fontsize=12)
fig.savefig('./images/paramsensitivity.png',dpi=400,bbox_inches='tight')


fig.savefig('./images/paramsensitivityunchanged.pdf',bbox_inches='tight')


#%%

x0 = np.array(mc.MacroscopicClass(alphaD=0.0,alphaS=1.0,lamb=0.32,beta=0,Eca=1e2,power=3))

paramtosweep = 9

modelstosweep = [2]
# modelstosweep = [0,1]

nsweep = 40
npoints = 8000


# paramvals = np.logspace(0,5,nsweep)
paramvals = np.logspace(-6,-1,nsweep)
params = []

βvals = np.linspace(0.01,0.1,10)
EcaGolf = np.flip(1/βvals)
paramgolf = []
for j in range(len(EcaGolf)):
    x = x0.copy()
    x[5] = EcaGolf[j]
    paramgolf.append(x)


for j in range(len(paramvals)):
    x = x0.copy()
    x[paramtosweep] = paramvals[j]
   
    params.append(x)






sims = []

for j in range(len(modelstosweep)):

    if modelstosweep[j] ==1 and paramtosweep==5:
        param = paramgolf
    else:
        param=params


    models = [modelstosweep[j]]*len(param)
    if models == 1:
        solver = ['mcnp']*len(param)
    else:
        solver = ['mc']*len(param)


    colors = sns.color_palette("deep",len(param)+1)

    legend = [0]*len(param)
    sims.append(simulation(npoints, param,legend,model= models,colors=colors,solver=solver)
                )

    fig,fabrics = sims[j].stream()


#%%
colors = sns.color_palette("deep",len(params)+1)

dmin = 550

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
#     "font.size" : 10,
    
# })


#label per param index
fig, ax = plt.subplots(1,len(modelstosweep),figsize=(4,3))



for j in range(len(modelstosweep)):

    dind = np.abs(sims[j].depths-dmin).argmin()
    dind_stoll = np.abs(sims[j].stoll_d-dmin).argmin()

    ev_nl = []
    ev_sl = []
    ev_zl = []

    ev_nl.append(sims[j].e_n[dind_stoll:])
    ev_sl.append(sims[j].e_s[dind_stoll:])
    ev_zl.append(sims[j].e_z[dind_stoll:])

    for i in range(len(sims[j].params)):
        ev_nl.append(sims[j].ev_n[i,:][dind:])

        ev_sl.append(sims[j].ev_s[i,:][dind:])
        ev_zl.append(sims[j].ev_z[i,:][dind:])

    # ax[0].violinplot(ev_nl,colors=colors)
    # ax[1].violinplot(woodcock)#,labels=['EGRIP']+legend)

    ev_nl_mean = np.zeros(len(ev_nl))
    ev_sl_mean = np.zeros(len(ev_nl))
    ev_zl_mean = np.zeros(len(ev_nl))

    ev_nl_std = np.zeros(len(ev_nl))
    ev_sl_std = np.zeros(len(ev_nl))
    ev_zl_std = np.zeros(len(ev_nl))

    if len(modelstosweep) == 1:
        a = ax
    else:
        a = ax[j]



    for i in range(len(ev_nl)):
        ev_nl_mean[i] = np.mean(ev_nl[i])
        ev_sl_mean[i] = np.mean(ev_sl[i])
        ev_zl_mean[i] = np.mean(ev_zl[i])

        ev_nl_std[i] = np.std(ev_nl[i])
        ev_sl_std[i] = np.std(ev_sl[i])
        ev_zl_std[i] = np.std(ev_zl[i])


    if modelstosweep[j] ==1 and paramtosweep ==5:
        pv = EcaGolf
    else:
        pv = paramvals

    if j == 0:
        a.plot(pv,ev_nl_mean[1:],color=colors[0],label='Largest Eigenvalue')
        a.plot(pv,ev_sl_mean[1:],color=colors[1],label='Smallest Eigenvalue')
        a.plot(pv,ev_zl_mean[1:],color=colors[2],label='Middle Eigenvalue')

        a.set_ylabel('Eigenvalue')

        # ax[j].axhline(ev_nl_mean[0],linestyle='--',color=colors[0],linewidth=0.75, label='EGRIP data')
        # ax[j].axhline(ev_sl_mean[0],linestyle='--',color=colors[1],linewidth=0.75)
        # ax[j].axhline(ev_zl_mean[0],linestyle='--',color=colors[2],linewidth=0.75)
    else:
        a.plot(pv,ev_nl_mean[1:],color=colors[0])
        a.plot(pv,ev_sl_mean[1:],color=colors[1])
        a.plot(pv,ev_zl_mean[1:],color=colors[2])

    
    # a.set_xlabel(mc.ModelNames(modelstosweep[j]))
    a.set_xlabel(mc.ParamNames(paramtosweep,latex=True))
    
    # axis title
    if len(modelstosweep) > 1:
        a.set_title(mc.ModelNames(modelstosweep[j]))



        # ax[j].axhline(ev_nl_mean[0],linestyle='--',color=colors[0],linewidth=0.75)
        # ax[j].axhline(ev_sl_mean[0],linestyle='--',color=colors[1],linewidth=0.75)
        # ax[j].axhline(ev_zl_mean[0],linestyle='--',color=colors[2],linewidth=0.75)



    a.set_ylim(0.0,1.0)
    #add grid
    a.grid(True,which='both',axis='both',linestyle='--',linewidth=0.5)
    if j>0:
        a.set_yticklabels([])
    a.locator_params(axis='x', nbins=5)
    if paramtosweep == 5 or paramtosweep > 7.5:
        a.set_xscale('log')


#add text at bottom showing other parameters
      
# get ind of all but paramtosweep
if paramtosweep < 8:
    ind = np.arange(7)
    ind = np.delete(ind,paramtosweep)
else:
    ind = np.arange(10)
    ind = np.delete(ind,7)
    ind = np.delete(ind,paramtosweep-1)

# format to 3 sig figs
formatspec = '{:.3g}'
text = ''
for i in ind:
    pi = x0[i]
    if i == 1:
        pi /= mc.iota(x0[4],x0[5])
    text += mc.ParamNames(i,latex=True) + ' = ' + formatspec.format(pi) + ', '

# strip final comma
text = text[:-2]
fig.text(0.5, -0.1,text, ha='center',fontsize=11)

     


paramname = mc.ParamNames(paramtosweep,latex=True)



# add fig title, moved up slightly

fig.suptitle('Parameter sensitivity for ' + paramname +': Average eigenvalues at EGRIP from ' +str(dmin)+' to ' +str(int(sims[0].depths[-1]))+' m',
             y=1.05,fontsize=12)

paramfilename = mc.ParamNames(paramtosweep,latex=False)

fig.savefig('./images/paramsensitivity'+paramfilename+'.pdf',bbox_inches='tight')
