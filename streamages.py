#%%
import numpy as np# numpy for arrays
from matplotlib import pyplot as plt
from simulations import simulation
import seaborn as sns
import mcfab as mc
import jax.numpy as jnp
plt.rcParams.update({
#    "text.usetex": True,
 #   "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 12,
    "figure.autolayout" : False,
    
})


ages = np.linspace(5,10000,40)
# ages = np.logspace(1,5,40)

npoints = 3000

x = mc.GrainClass(alpha=1.0,lamb=0.32,Eca=1e2,power=3)

params = [x]*len(ages)
model = [0]*len(ages)

lev = 0.75
sev = (1-lev)/2
a2 = jnp.array([[sev,0,0],[0,sev,0],[0,0,lev]])
a4 = mc.closure.IBOFClosure(a2)
n0,m0 = mc.build_discrete(a2,a4,npoints)



colors = sns.color_palette("viridis",len(ages))
## add black as first colour
colors.insert(0,'k')

legend = [f"Age = {age} yrs" for age in ages]

sim = simulation(npoints, params, legend, model,colors=colors,fabric0=(n0,m0))

fig,fabrics = sim.stream(ages)

#%%

colors = sns.color_palette("deep",3)
dmin = 550
dind = np.abs(sim.depths-dmin).argmin()
dind_stoll = np.abs(sim.stoll_d-dmin).argmin()

fig,ax = plt.subplots(1,1,figsize=(7,4))

scatter_kwargs = {'marker': 'x', \
                                 's': 50,\
}



meanevs = np.zeros((len(sim.params),3))
meanevs2 = np.zeros((len(sim.params),3))
for i in range(len(sim.params)):
    meanevs[i,0] = np.mean(sim.ev_n[i,:][dind:])
    meanevs[i,1] = np.mean(sim.ev_s[i,:][dind:])
    meanevs[i,2] = np.mean(sim.ev_z[i,:][dind:])
    # meanevs2[i,0] = np.mean(sim2.ev_n[i,:][dind:])
    # meanevs2[i,1] = np.mean(sim2.ev_s[i,:][dind:])
    # meanevs2[i,2] = np.mean(sim2.ev_z[i,:][dind:])


ax.plot(ages,meanevs[:,0],color=colors[0])
ax.plot(ages,meanevs[:,1],color=colors[1])
ax.plot(ages,meanevs[:,2],color=colors[2])

# ax.plot(ages,meanevs2[:,0],color=colors[0])
# ax.plot(ages,meanevs2[:,1],color=colors[1])
# ax.plot(ages,meanevs2[:,2],color=colors[2])
# add horizontal line at EGRIP value including std dev

std_n = np.std(sim.e_n[dind_stoll:])
std_s = np.std(sim.e_s[dind_stoll:])
std_z = np.std(sim.e_z[dind_stoll:])

ax.axhline(y=np.mean(sim.e_n[dind_stoll:]),color=colors[0],linestyle='--',alpha=0.6)
ax.axhline(y=np.mean(sim.e_s[dind_stoll:]),color=colors[1],linestyle='--',alpha=0.6)
ax.axhline(y=np.mean(sim.e_z[dind_stoll:]),color=colors[2],linestyle='--',alpha=0.6)

ax.axhspan(np.mean(sim.e_n[dind_stoll:])-std_n,np.mean(sim.e_n[dind_stoll:])+std_n,color=colors[0],alpha=0.2)
ax.axhspan(np.mean(sim.e_s[dind_stoll:])-std_s,np.mean(sim.e_s[dind_stoll:])+std_s,color=colors[1],alpha=0.2)
ax.axhspan(np.mean(sim.e_z[dind_stoll:])-std_z,np.mean(sim.e_z[dind_stoll:])+std_z,color=colors[2],alpha=0.2)


ax.set_ylabel('Eigenvalues')
ax.set_ylim([-0.05,0.9])

ax.set_xlabel('Age of NEGIS (yrs)')
# ax.set_xscale('log')

# grid
ax.grid(True,which='both',linestyle='--',linewidth=0.5)

# y log
# ax.set_yscale('log')
fig.suptitle('Mean eigenvalues at EGRIP, '+str(dmin)+' to ' +str(int(sim.depths[-1]))+' m')


#Add custom legend to include EGRIP horizontal line
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
patch = Patch(color='k',alpha=0.2)
line = Line2D([0], [0], color='k', lw=2,linestyle='--',alpha=0.6)
line2 = Line2D([0], [0], color='k', lw=2)
plt.legend([(patch,line),line2],['EGRIP','Model'])

fig.savefig('./images/streamage.png', bbox_inches='tight',dpi=500)
fig.savefig('./images/streamage.pdf', bbox_inches='tight')
