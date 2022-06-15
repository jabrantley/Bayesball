########################################################################
#                                                                      #
#                MAKE SUBPLOTS FOR FIGURE 1 - R.A. Dickey              #
#                                                                      #
########################################################################

# The v3 figure 1 is made in inkscape, but the data figures are made here

# Pitches and order_dickey
pitches_dickey = ["4-Seam Fastball","Knuckleball"]   
order_addKN = pd.concat([order_greinke,pd.DataFrame({"pitch": ["Knuckleball"], "color": ["black"],"marker": ["v"], "code": [pitch_type_codes["Knuckleball"]]})],ignore_index=True, axis=0) 
order_dickey = order_addKN.loc[order_addKN.pitch.isin(pitches_dickey),:]
order_dickey.reset_index(inplace=True)

# Get data for Tyler Glasnow
dickey  = data.loc[data.player_name.isin(["Dickey, R.A."]) & data.pitch_name.isin(pitches_dickey),:].copy() #[2018, 2019, 2020]

# dickey = data[data.player_name.isin(["dickey, Zack"])].copy()

# dickey["release_spin_rate"] = dickey["release_spin_rate"].fillna(0)
dickey.loc[:,"release_speed_float"] = dickey.release_speed.astype("float64")
dickey.loc[:,"release_spin_float"] = dickey.release_spin_rate.astype("float64")
dickey.loc[:,"release_pos_z_float"] = dickey.release_pos_z.astype("float64")
dickey.loc[:,"pfx_z_float"] = dickey.pfx_z.astype("float64")
dickey.loc[:,"plate_z_float"] = dickey.plate_z.astype("float64")
dickey.loc[:,"plate_z_norm"] = np.array((dickey.plate_z - dickey.sz_bot) / (dickey.sz_top - dickey.sz_bot), dtype="float")

prior_color = "dodgerblue"
likelihood_color = (0.5,0.5,0.5)
posterior_color = order_greinke[order_greinke.pitch.isin(["Curveball"])].color.tolist()[0]
MAPcolor = order_greinke[order_greinke.pitch.isin(["4-Seam Fastball"])].color.tolist()[0]

#########################################
#                                       #
#       Make full bayes for batter      #
#                                       #
#########################################

# Get pdf for dickey data - aka prior
mean,std=stats.norm.fit(dickey.loc[dickey.pitch_name.isin(pitches_dickey),"plate_z_norm"])
print(mean,std)
# mean,std=stats.norm.fit(dickey.loc[dickey.pitch_name.isin(["4-Seam Fastball"]),"plate_z_norm"])#dickey.plate_z_norm)
X= np.linspace(-.5, 1.5, 100)
dickey_prior_fastball =   stats.norm.pdf(X, mean, std) #
dickey_prior_fastball /= np.sum(dickey_prior_fastball)
# Get likelihood
likelihood_fastball = stats.norm.pdf(X, 0.75, 0.2)
likelihood_fastball /= np.sum(likelihood_fastball)
# Get posterior
posterior_fastball = np.multiply(dickey_prior_fastball,likelihood_fastball)
posterior_fastball /= np.sum(posterior_fastball)

# Get pdf for dickey data - aka prior
mean,std=stats.norm.fit(dickey.loc[dickey.pitch_name.isin(["Knuckleball"]),"plate_z_norm"])#dickey.plate_z_norm)
X= np.linspace(-.5, 1.5, 100)
dickey_prior_knuckleball_true = stats.norm.pdf(X, mean,std) #stats.norm.pdf(X, mean, std) #stats.norm.pdf(X, 0.5, 0.5) #
dickey_prior_knuckleball_true /= np.sum(dickey_prior_knuckleball_true)
dickey_prior_knuckleball = stats.norm.pdf(X, 0.5, 0.6) #stats.norm.pdf(X, mean, std) #stats.norm.pdf(X, 0.5, 0.5) #
dickey_prior_knuckleball /= np.sum(dickey_prior_knuckleball)
# Get likelihood
likelihood_knuckleball = stats.norm.pdf(X, 0.25, 0.3)
likelihood_knuckleball /= np.sum(likelihood_knuckleball)
# Get posterior
posterior_knuckleball = np.multiply(dickey_prior_knuckleball,likelihood_knuckleball)
posterior_knuckleball /= np.sum(posterior_knuckleball)

likelihood_eephus = stats.norm.pdf(X, 0.25, 0.1)
likelihood_eephus /= np.sum(likelihood_eephus)
# Get posterior
posterior_eephus = np.roll(likelihood_eephus,1) #np.multiply(dickey_prior_knuckleball,likelihood_knuckleball)

####################
#                  #
#     Fastball     # 
#                  #
####################

# Make figure
fig = plt.figure(figsize=(1.5,4))
ax = fig.add_subplot()
ax.plot(dickey_prior_fastball,X,color=prior_color,zorder=2,label="Prior")
# ax.fill_between(dickey_prior,X,color="dodgerblue",zorder=2,alpha=0.2)
ax.plot(likelihood_fastball,X,color=likelihood_color,zorder=3,label="Likelihood")
# ax.fill_between(likelihood,X,color=(0.5,0.5,0.5),zorder=3,alpha=0.2)
ax.plot(posterior_fastball,X,color=posterior_color,zorder=4,label="Posterior")
# ax.fill_between(posterior,X,color=order_dickey[order_dickey.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,alpha=0.2)
ax.scatter(posterior_fastball.max(),X[posterior_fastball.argmax()],s=50,color=MAPcolor,zorder=5,marker="*",label="Estimate")

# Make figure adjustments
ax.set_ylim(-.5,1.5)
bin_ticks=[-.5,0,0.25,0.5,0.75,1,1.5]
ax.set_yticks([0,1])#ticks=bin_ticks)
# ax.set_yticklabels(list(map(str, [round(val,2) for val in bin_ticks])))#['1','2','3','4','5','6','7','8'])
ax.tick_params(axis='y')#,direction='out')
ax.set_ylabel(None)
ax.xaxis.set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Dickey","Figure1-v2-PriorLikelihoodPosterior-DickeyFastball.svg"))

####################
#                  #
#    Knuckleball   # 
#                  #
####################
# Make figure
fig = plt.figure(figsize=(1.5,4))
ax = fig.add_subplot()
ax.plot(dickey_prior_knuckleball_true,X,color=prior_color,zorder=2,label="Prior")
ax.plot(dickey_prior_knuckleball,X,color=prior_color,linestyle='--',zorder=2,label="Prior")
# ax.fill_between(dickey_prior,X,color="dodgerblue",zorder=2,alpha=0.2)
ax.plot(likelihood_knuckleball,X,color=likelihood_color,zorder=3,label="Likelihood")
# ax.fill_between(likelihood,X,color=(0.5,0.5,0.5),zorder=3,alpha=0.2)
ax.plot(posterior_knuckleball,X,color=posterior_color,zorder=4,label="Posterior")
# ax.fill_between(posterior,X,color=order_dickey[order_dickey.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,alpha=0.2)
ax.scatter(posterior_knuckleball.max(),X[posterior_knuckleball.argmax()],s=50,color=MAPcolor,zorder=5,marker="*",label="Estimate")

# Make figure adjustments
ax.set_ylim(-.5,1.5)
bin_ticks=[-.5,0,0.25,0.5,0.75,1,1.5]
ax.set_yticks([0,1])#ticks=bin_ticks)
# ax.set_yticklabels(list(map(str, [round(val,2) for val in bin_ticks])))#['1','2','3','4','5','6','7','8'])
ax.tick_params(axis='y')#,direction='out')
ax.set_ylabel(None)
ax.xaxis.set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Dickey","Figure1-v2-PriorLikelihoodPosterior-DickeyKnuckleball.svg"))


####################
#                  #
#      Eephus      # 
#                  #
####################
# Make figure
fig = plt.figure(figsize=(1.5,4))
ax = fig.add_subplot()
# ax.plot(dickey_prior_fastball,X,color=prior_color,zorder=2,label="Prior")
ax.plot(dickey_prior_knuckleball_true,X,color=prior_color,linestyle='--',zorder=2,label="Prior")
# ax.fill_between(dickey_prior,X,color="dodgerblue",zorder=2,alpha=0.2)
ax.plot(likelihood_eephus,X,color=likelihood_color,zorder=3,label="Likelihood")
# ax.fill_between(likelihood,X,color=(0.5,0.5,0.5),zorder=3,alpha=0.2)
ax.plot(posterior_eephus,X,color=posterior_color,zorder=4,label="Posterior")
# ax.fill_between(posterior,X,color=order_dickey[order_dickey.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,alpha=0.2)
ax.scatter(posterior_eephus.max(),X[posterior_eephus.argmax()],s=50,color=MAPcolor,zorder=5,marker="*",label="Estimate")

# Make figure adjustments
ax.set_ylim(-.5,1.5)
bin_ticks=[-.5,0,0.25,0.5,0.75,1,1.5]
ax.set_yticks([0,1])#ticks=bin_ticks)
# ax.set_yticklabels(list(map(str, [round(val,2) for val in bin_ticks])))#['1','2','3','4','5','6','7','8'])
ax.tick_params(axis='y')#,direction='out')
ax.set_ylabel(None)
ax.xaxis.set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Dickey","Figure1-v2-PriorLikelihoodPosterior-DickeyEephus.svg"))




#########################################
#                                       #
#           Make P(z| pitch)            #
#                                       #
#########################################

# Make figure
fig = plt.figure(figsize=(0.75,3.5))
ax = fig.add_subplot()

for pitch in pitches_dickey:
    # Get pdf for dickey data - aka prior
    mean,std=stats.norm.fit(dickey.loc[dickey.pitch_name.isin([pitch]),"plate_z_norm"])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std)
    ax.plot(prior,X,color=order_dickey.loc[order_dickey.pitch.isin([pitch]),"color"].tolist()[0],label="Prior")

# Make figure adjustments
ax.set_ylim(-.5,1.5)
ax.set_yticks([0,1])
ax.set_yticklabels([])
ax.tick_params(axis='y')
ax.set_ylabel(None)
ax.xaxis.set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Dickey","Figure1-v3-PlatePositionByPitch-Dickey.svg"))

#########################################
#                                       #
#             Make P(z|velo)            #
#                                       #
#########################################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

dickey.dropna(subset=['release_speed'],inplace=True)
speed_bins = [dickey.release_speed.min(), 75, dickey.release_speed.max()]
dickey.loc[:,'release_speed_binned'], retbins = pd.cut(dickey['release_speed'].astype("float"),bins=speed_bins,retbins=True)

clrs = ['black','grey']
# for cnt,(bin,clr,lnstyle) in enumerate(zip(dickey.loc[:,'release_speed_binned'].unique(),['black','grey'],['-',':'])):
for cnt,(bin,clr,lnstyle) in enumerate(zip(dickey.loc[:,'release_speed_binned'].unique().sort_values(),['black','grey'],['-','--'])):
    # Get pdf for dickey data - aka prior
    mean,std=stats.norm.fit(dickey.loc[dickey.release_speed_binned == bin,'plate_z_norm'])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std) / len(dickey.loc[dickey.release_speed_binned == bin,'plate_z_norm'])
    ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} mph".format(bin.left,bin.right))

# bin_labels = list(map(str, range(1,5))) #['1','2','3','4','5','6','7','8','9']
# velo_qbinned , velo_qbins = pd.qcut(dickey['release_speed'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
# dickey.loc[:,'release_speed_qbinned'] = velo_qbinned

# clrs = ['black','darkgrey','grey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(dickey.loc[:,'release_speed_binned'].unique(),['black','grey'],['-',':'])):
# for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','darkgrey'],['-',':','--'])):
#     # Get pdf for dickey data - aka prior
#     mean,std=stats.norm.fit(dickey.loc[dickey.release_speed_qbinned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} mph".format(velo_qbins[cnt],velo_qbins[cnt+1]))

# Make figure adjustments
ax.set_ylim(-.5,1.5)
ax.set_yticks([0,1])
ax.set_yticklabels([])
ax.tick_params(axis='y')
ax.set_ylabel(None)
ax.xaxis.set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Dickey","Figure1-v3-PlatePositionByVelo-Dickey.svg"))

#########################################
#                                       #
#              Make P(z|spin)           #
#                                       #
#########################################

# # Make figure
# fig = plt.figure(figsize=(1,3.5))
# ax = fig.add_subplot()

# dickey = dickey.dropna(subset=['spin_axis'])

# # spin_cutbins = [0,150,360]
# # dickey.loc[:,'spin_axis_binned'] = pd.cut(dickey['spin_axis'].astype("float"),bins=spin_cutbins)

# # clrs = ['black','darkgrey','grey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(dickey.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
# #     # Get pdf for dickey data - aka prior
# #     mean,std=stats.norm.fit(dickey.loc[dickey.spin_axis_binned == bin,'plate_z_norm'])
# #     X= np.linspace(-.5, 1.5, 100)
# #     prior = stats.norm.pdf(X, mean, std)
# #     ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} rpm".format(spin_cutbins[cnt],spin_cutbins[cnt+1]))

# bin_labels = list(map(str, range(1,6))) #['1','2','3','4','5','6','7','8','9']
# spin_qbinned , spin_qbins = pd.qcut(dickey['spin_axis'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
# dickey.loc[:,'spin_axis_qbinned'] = spin_qbinned

# clrs = ['black','darkgrey','grey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(dickey.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
# for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','lightgrey'],['-','--',':'])):
#     # Get pdf for dickey data - aka prior
#     mean,std=stats.norm.fit(dickey.loc[dickey.spin_axis_qbinned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{}".format(spin_qbins[cnt],spin_qbins[cnt+1]))

# # Make figure adjustments
# ax.set_ylim(-.5,1.5)
# ax.set_yticks([0,1])
# ax.set_yticklabels([])
# ax.tick_params(axis='y')
# ax.set_ylabel(None)
# ax.xaxis.set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.legend()

# fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Dickey","Figure1-v3-PlatePositionBySpinAxis-Dickey.svg"))


#################
#               #
#   Spin Rate   #
#               #
#################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

dickey.loc[:,"release_spin_rate"] = dickey.release_spin_rate.fillna(0.1)
# dickey = dickey.dropna(subset=['release_spin_rate'])

spin_cutbins = [0,300,1000,1400,dickey.release_spin_rate.max()]
dickey.loc[:,'release_spin_rate_binned'] = pd.cut(dickey['release_spin_rate'].astype("float"),bins=spin_cutbins)

# bin_labels = list(map(str, range(1,4))) #['1','2','3','4','5','6','7','8','9']
# spinrate_qbinned , spinrate_qbins = pd.qcut(dickey['release_spin_rate'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
# dickey.loc[:,'release_spin_rate_qbinned'] = spinrate_qbinned

clrs = ['black','darkgrey','grey', 'grey']
# for cnt,(bin,clr,lnstyle) in enumerate(zip(dickey.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
for cnt,(bin,clr,lnstyle) in enumerate(zip(dickey.loc[:,'release_spin_rate_binned'].unique().sort_values(),['black','grey','darkgrey','lightgrey'],['-','--','-.',':'])):
    if len(dickey.loc[dickey.release_spin_rate_binned == bin,'plate_z_norm']) > 20:
        # Get pdf for dickey data - aka prior
        mean,std=stats.norm.fit(dickey.loc[dickey.release_spin_rate_binned == bin,'plate_z_norm'])
        X= np.linspace(-.5, 1.5, 100)
        prior = stats.norm.pdf(X, mean, std) / len(dickey.loc[dickey.release_spin_rate_binned == bin,'plate_z_norm'])
        ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} rpm".format(bin.left,bin.right))

# Make figure adjustments
ax.set_ylim(-.5,1.5)
ax.set_yticks([0,1])
ax.set_yticklabels([])
ax.tick_params(axis='y')
ax.set_ylabel(None)
ax.xaxis.set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Dickey","Figure1-v3-PlatePositionBySpinRate-Dickey.svg"))


#########################
#                       #
#   Vertical Movement   #
#                       #
#########################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

dickey = dickey.dropna(subset=['pfx_z_norm'])

pfxz_cutbins = [dickey['pfx_z_norm'].min(), 0 , dickey['pfx_z_norm'].max()]
dickey.loc[:,'pfx_z_binned'] = pd.cut(dickey['pfx_z_norm'].astype("float"),bins=pfxz_cutbins)

clrs = ['black','grey']
for cnt,(bin,clr,lnstyle) in enumerate(zip(dickey.loc[:,'pfx_z_binned'].unique(),['black','grey'],['-','--'])):
    # Get pdf for dickey data - aka prior
    mean,std=stats.norm.fit(dickey.loc[dickey.pfx_z_binned == bin,'plate_z_norm'])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std) / len(dickey.loc[dickey.pfx_z_binned == bin,'plate_z_norm'])
    ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} inches".format(pfxz_cutbins[cnt],pfxz_cutbins[cnt+1]))

# bin_labels = list(map(str, range(1,5))) #['1','2','3','4','5','6','7','8','9']
# pfxz_qbinned , pfxz_qbins = pd.qcut(dickey['pfx_z_float'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
# dickey.loc[:,'pfx_z_qbinned'] = pfxz_qbinned

# clrs = ['black','darkgrey','grey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(dickey.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
# for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','lightgrey'],['-','--',':'])):
#     # Get pdf for dickey data - aka prior
#     mean,std=stats.norm.fit(dickey.loc[dickey.pfx_z_qbinned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} inches".format(pfxz_qbins[cnt],pfxz_qbins[cnt+1]))

# Make figure adjustments
ax.set_ylim(-.5,1.5)
ax.set_yticks([0,1])
ax.set_yticklabels([])
ax.tick_params(axis='y')
ax.set_ylabel(None)
ax.xaxis.set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Dickey","Figure1-v3-PlatePositionByVerticalMovement-Dickey.svg"))



# #########################
# #                       #
# #   Vertical Movement   #
# #                       #
# #########################

# # Make figure
# fig = plt.figure(figsize=(1,3.5))
# ax = fig.add_subplot()

# dickey.loc[:,"vertical_drop"] = dickey.loc[:,"release_pos_z_float"] - dickey.loc[:,"plate_z"]
# dickey.dropna(subset=['vertical_drop'],inplace=True)

# # vdrop_cutbins = [dickey['vertical_drop'].min(), 0 , dickey['pfx_z_float'].max()]
# # dickey.loc[:,'pfx_z_binned'] = pd.cut(dickey['pfx_z_float'].astype("float"),bins=pfxz_cutbins)

# # clrs = ['black','darkgrey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(dickey.loc[:,'pfx_z_binned'].unique(),['black','grey'],['-',':'])):
# #     # Get pdf for dickey data - aka prior
# #     mean,std=stats.norm.fit(dickey.loc[dickey.pfx_z_binned == bin,'plate_z_norm'])
# #     X= np.linspace(-.5, 1.5, 100)
# #     prior = stats.norm.pdf(X, mean, std)
# #     ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} inches".format(pfxz_cutbins[cnt],pfxz_cutbins[cnt+1]))

# bin_labels = list(map(str, range(1,5))) #['1','2','3','4','5','6','7','8','9']
# vdrop_qbinned , vdrop_qbins = pd.qcut(dickey['vertical_drop'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
# dickey.loc[:,'vertical_drop_qbinned'] = vdrop_qbinned

# clrs = ['black','darkgrey','grey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(dickey.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
# for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','lightgrey','darkgrey'],['-','--',':','-.'])):
#     # Get pdf for dickey data - aka prior
#     mean,std=stats.norm.fit(dickey.loc[dickey.vertical_drop_qbinned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} ft".format(round(vdrop_qbins[cnt],1),round(vdrop_qbins[cnt+1],1)))

# # Make figure adjustments
# ax.set_ylim(-.5,1.5)
# ax.set_yticks([0,1])
# ax.set_yticklabels([])
# ax.tick_params(axis='y')
# ax.set_ylabel(None)
# ax.xaxis.set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.legend()

# fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Dickey","Figure1-v3-PlatePositionByVerticalDrop-Dickey.svg"))

#########################################
#                                       #
#      Make scatter of spin vs velo     #
#                                       #
#########################################

fig = plt.figure(figsize=(2,1))
ax = fig.add_subplot()
ss = sns.scatterplot(data=dickey.loc[dickey.pitch_name.isin(pitches_dickey),:],y='release_speed_float', x='release_spin_rate',hue="pitch_name",hue_order=order_dickey.pitch.tolist(),palette=order_dickey.color.tolist(),legend=False,size=2,linewidth=.1)
ax.set_xlabel("Spin rate (rpm)")
ax.set_ylabel("Release velocity (mph)")
ax.set_title("Inferred velocity from spin")
plt.rcParams.update({'font.sans-serif':'Arial','font.size':8})
ax.set_yticks([80,90,100])
ax.set_ylim([75,105])
fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Dickey","Figure1-v3-SpinRateVsVelocity-Dickey.svg"))

#########################################
#                                       #
#      Make P(launch angle | pitch)     #
#                                       #
#########################################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

for pitch in pitches_dickey:
    # Get pdf for dickey data - aka prior
    mean,std=stats.norm.fit(dickey.loc[dickey.pitch_name.isin([pitch]),"launch_angle"])
    X= np.linspace(-100, 100, 100)
    _P = stats.norm.pdf(X, mean, std)
    ax.plot(_P,X,color=order_dickey.loc[order_dickey.pitch.isin([pitch]),"color"].tolist()[0])

# Make figure adjustments
# ax.set_ylim(-.5,1.5)
# ax.set_yticks([0,1])
# ax.set_yticklabels([])
ax.tick_params(axis='y')
ax.set_ylabel("Degrees")
ax.xaxis.set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Dickey","Figure1-v3-LaunchAngleByPitch-Dickey.svg"))

##########################################################
#                                                        #
#      Make scatter of launch angle vs plate position    #
#                                                        #
##########################################################

# Make violin instead
# my_violinplot(local_data=dickey,
#               true_shift="plate_z_norm",
#               deviation="launch_angle",
#               ax=ax,
#               which_pitches_dickey=pitches_dickey,
#               bin_locs=which_bins_mean)

fig = plt.figure(figsize=(2,1))
ax = fig.add_subplot()

# For each bin, plot as jittered scatter plot
for (bin, bin_loc) in zip(sorted(dickey.plate_z_norm_qbinned.unique()),which_bins_mean):
    _yvals = dickey.loc[dickey.pitch_name.isin(pitches_dickey) & (dickey.plate_z_norm_qbinned==bin),:].launch_angle
    xwithnoise = len(_yvals)*[bin_loc] + np.random.normal(0,.001,len(_yvals))
    ax.scatter(xwithnoise,_yvals,s=5,color="lightgray")

# For each pitch type
thisx = np.linspace(-.65,1.65, 100)
# each_pitch_ols = []
# for row in order_dickey.itertuples():
#     if (len(dickey.loc[dickey.pitch_name.isin([row.pitch])]) >= 100):
#         ols_result = run_regression(dickey.loc[dickey.pitch_name.isin([row.pitch])].plate_z_norm,dickey.loc[dickey.pitch_name.isin([row.pitch])].launch_angle)
#         ax.plot(thisx,ols_result.params[0] + (ols_result.params[1]*thisx),linewidth=2,linestyle=":",color=row.color,label=row.pitch)

# Regression line for all pitches_dickey
ols_result = run_regression(dickey.loc[dickey.pitch_name.isin(pitches_dickey)].plate_z_norm,dickey.loc[dickey.pitch_name.isin(pitches_dickey)].launch_angle)
ax.plot(thisx,ols_result.params[0] + (ols_result.params[1]*thisx),linewidth=2, color=MAPcolor,label="All pitches_dickey")

# Clean up
ax.set_yticks([-100,-50,0,50,100])
ax.set_xticks([0,1])
ax.set_xticklabels(["Knees","Chest"],color="firebrick",fontweight="bold")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.rcParams.update({'font.sans-serif':'Arial','font.size':6})
ax.set_xlabel("Vertical Plate Position (% strike zone)",fontweight="bold",fontsize=6)
ax.set_ylabel("Launch Angle (deg)",fontweight="bold",fontsize=6)


fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Dickey","Figure1-v3-LaunchAngleVersusPlateZ-Dickey.svg"))

plt.rcParams.update({'font.sans-serif':'Arial','font.size':10})

#########################################
#                                       #
#           Make P(pitch)            #
#                                       #
#########################################

# Make figure
fig = plt.figure(figsize=(2,2.5))
ax = fig.add_subplot()

countplot=sns.histplot(data=data.loc[data.player_name.isin(["Dickey, R.A."]),:],y="player_name",hue="pitch_name",hue_order=data.loc[data.player_name.isin(["Dickey, R.A."]),:].pitch_name.value_counts(sort=True).index.tolist(),palette=[order_dickey.loc[order_dickey.pitch.isin(["4-Seam Fastball"]),"color"].item(),'k',tuple(np.divide([86,180,233],255))], multiple="dodge", 
              stat='density', shrink=0.8, common_norm=True)


ax.legend_.remove()
# ax.set_yticks([])
ax.set_xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.set_visible(False)
ax.patch.set_alpha(0)
ax.set_title("Pitch distribution",fontsize=10)
ax.set(xlabel=None,ylabel=None)

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Dickey","Figure1-v3-PitchDistribution-Dickey.svg"))
