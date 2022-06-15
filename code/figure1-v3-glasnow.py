########################################################################
#                                                                      #
#                MAKE SUBPLOTS FOR FIGURE 1 - Tyler Glasnow            #
#                                                                      #
########################################################################

# The v3 figure 1 is made in inkscape, but the data figures are made here

# Pitches and order_glasnow
pitches_glasnow = ["4-Seam Fastball","Curveball","Slider","Changeup"]    
order_glasnow = order_greinke.loc[order_greinke.pitch.isin(pitches_glasnow),:]
order_glasnow.reset_index(inplace=True)

# Get data for Tyler Glasnow
glasnow  = data.loc[data.game_year.isin([2018, 2019, 2020]) & data.player_name.isin(["Glasnow, Tyler"]) & data.pitch_name.isin(pitches_glasnow),:].copy()#[2018, 2019, 2020]

# glasnow = data[data.player_name.isin(["glasnow, Zack"])].copy()

# glasnow["release_spin_rate"] = glasnow["release_spin_rate"].fillna(0)
glasnow.loc[:,"release_speed_float"] = glasnow.release_speed.astype("float64")
glasnow.loc[:,"release_spin_float"] = glasnow.release_spin_rate.astype("float64")
glasnow.loc[:,"release_pos_z_float"] = glasnow.release_pos_z.astype("float64")
glasnow.loc[:,"pfx_z_float"] = glasnow.pfx_z.astype("float64")
glasnow.loc[:,"plate_z_float"] = glasnow.plate_z.astype("float64")
glasnow.loc[:,"plate_z_norm"] = np.array((glasnow.plate_z - glasnow.sz_bot) / (glasnow.sz_top - glasnow.sz_bot), dtype="float")

glasnow_alds = glasnow.loc[glasnow.game_date.isin(["2019-10-10"]),:]

#########################################
#                                       #
#       Make full bayes for batter      #
#                                       #
#########################################

# Get pdf for glasnow data - aka prior
mean_fb,std_fb=stats.norm.fit(glasnow.loc[glasnow.pitch_name.isin(["4-Seam Fastball"]),"plate_z_norm"])#glasnow.plate_z_norm)
X= np.linspace(-.5, 1.5, 100)
glasnow_prior_fastball = stats.norm.pdf(X, mean_fb, std_fb)
glasnow_prior_fastball /= np.sum(glasnow_prior_fastball)
# Get likelihood
likelihood_fastball = stats.norm.pdf(X, 0.75, 0.2)
likelihood_fastball /= np.sum(likelihood_fastball)
# Get posterior
posterior_fastball = np.multiply(glasnow_prior_fastball,likelihood_fastball)
posterior_fastball /= np.sum(posterior_fastball)

# Get pdf for glasnow data - aka prior
mean_cb,std_cb=stats.norm.fit(glasnow.loc[glasnow.pitch_name.isin(["Curveball"]),"plate_z_norm"])#glasnow.plate_z_norm)
X= np.linspace(-.5, 1.5, 100)
glasnow_prior_curveball = stats.norm.pdf(X, mean_cb, std_cb)
glasnow_prior_curveball /= np.sum(glasnow_prior_curveball)
# Get likelihood
likelihood_curveball = stats.norm.pdf(X, 0.25, 0.2) 
likelihood_curveball /= np.sum(likelihood_curveball )
# Get posterior
posterior_curveball = np.multiply(glasnow_prior_curveball,likelihood_curveball)
posterior_curveball /= np.sum(posterior_curveball)
####################
#                  #
#     Fastball     # 
#                  #
####################

# Make figure
fig = plt.figure(figsize=(1.5,4))
ax = fig.add_subplot()
ax.plot(glasnow_prior_fastball,X,color="dodgerblue",zorder=2,label="Prior")
# ax.fill_between(glasnow_prior,X,color="dodgerblue",zorder=2,alpha=0.2)
ax.plot(likelihood_fastball,X,color=(0.5,0.5,0.5),zorder=3,label="Likelihood")
# ax.fill_between(likelihood,X,color=(0.5,0.5,0.5),zorder=3,alpha=0.2)
ax.plot(posterior_fastball,X,color=order_glasnow[order_glasnow.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,label="Posterior")
# ax.fill_between(posterior,X,color=order_glasnow[order_glasnow.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,alpha=0.2)
ax.scatter(posterior_fastball.max(),X[posterior_fastball.argmax()],s=50,color=order_glasnow[order_glasnow.pitch.isin(["4-Seam Fastball"])].color.tolist()[0],zorder=5,marker="*",label="Estimate")

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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Glasnow","Figure1-v2-PriorLikelihoodPosterior-GlasnowFastball.svg"))

####################
#                  #
#    Curveball     # 
#                  #
####################
# Make figure
fig = plt.figure(figsize=(1.5,4))
ax = fig.add_subplot()
ax.plot(glasnow_prior_curveball,X,color="dodgerblue",zorder=2,label="Prior")
# ax.fill_between(glasnow_prior,X,color="dodgerblue",zorder=2,alpha=0.2)
ax.plot(likelihood_curveball,X,color=(0.5,0.5,0.5),zorder=3,label="Likelihood")
# ax.fill_between(likelihood,X,color=(0.5,0.5,0.5),zorder=3,alpha=0.2)
ax.plot(posterior_curveball,X,color=order_glasnow[order_glasnow.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,label="Posterior")
# ax.fill_between(posterior,X,color=order_glasnow[order_glasnow.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,alpha=0.2)
ax.scatter(posterior_curveball.max(),X[posterior_curveball.argmax()],s=50,color=order_glasnow[order_glasnow.pitch.isin(["4-Seam Fastball"])].color.tolist()[0],zorder=5,marker="*",label="Estimate")

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
fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Glasnow","Figure1-v2-PriorLikelihoodPosterior-GlasnowCurveball.svg"))

####################################
#                                  #
#    Sit on fastball get curve     # 
#                                  #
####################################
# Make figure
fig = plt.figure(figsize=(1.5,4))
ax = fig.add_subplot()
ax.plot(glasnow_prior_fastball,X,color="dodgerblue",zorder=2,label="Prior")
# ax.fill_between(glasnow_prior,X,color="dodgerblue",zorder=2,alpha=0.2)
ax.plot(likelihood_curveball,X,color=(0.5,0.5,0.5),zorder=3,label="Likelihood")
# ax.fill_between(likelihood,X,color=(0.5,0.5,0.5),zorder=3,alpha=0.2)
_P_mismatch = np.multiply(glasnow_prior_fastball,likelihood_curveball)
_P_mismatch /= np.sum(_P_mismatch)
ax.plot(_P_mismatch,X,color=order_glasnow[order_glasnow.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,label="Posterior")
# ax.fill_between(posterior,X,color=order_glasnow[order_glasnow.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,alpha=0.2)
ax.scatter(_P_mismatch.max(),X[_P_mismatch.argmax()],s=50,color=order_glasnow[order_glasnow.pitch.isin(["4-Seam Fastball"])].color.tolist()[0],zorder=5,marker="*",label="Estimate")

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
fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Glasnow","Figure1-v2-PriorLikelihoodPosterior-GlasnowNoTippingFastballPrior.svg"))


####################################
#                                  #
#    Sit on curve get fastball     # 
#                                  #
####################################
# Make figure
fig = plt.figure(figsize=(1.5,4))
ax = fig.add_subplot()
ax.plot(glasnow_prior_curveball,X,color="dodgerblue",zorder=2,label="Prior")
# ax.fill_between(glasnow_prior,X,color="dodgerblue",zorder=2,alpha=0.2)
ax.plot(likelihood_fastball,X,color=(0.5,0.5,0.5),zorder=3,label="Likelihood")
# ax.fill_between(likelihood,X,color=(0.5,0.5,0.5),zorder=3,alpha=0.2)
_P_mismatch = np.multiply(glasnow_prior_curveball,likelihood_fastball)
_P_mismatch /= np.sum(_P_mismatch)
ax.plot(_P_mismatch,X,color=order_glasnow[order_glasnow.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,label="Posterior")
# ax.fill_between(posterior,X,color=order_glasnow[order_glasnow.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,alpha=0.2)
ax.scatter(_P_mismatch.max(),X[_P_mismatch.argmax()],s=50,color=order_glasnow[order_glasnow.pitch.isin(["4-Seam Fastball"])].color.tolist()[0],zorder=5,marker="*",label="Estimate")

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
fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Glasnow","Figure1-v2-PriorLikelihoodPosterior-GlasnowNoTippingCurvePrior.svg"))


#########################################
#                                       #
#           Make P(z| pitch)            #
#                                       #
#########################################


# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

for pitch in ["4-Seam Fastball","Curveball"]:
    # Get pdf for glasnow data - aka prior
    mean,std=stats.norm.fit(glasnow.loc[glasnow.pitch_name.isin([pitch]),"plate_z_norm"])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std)
    # prior /= len(glasnow.loc[glasnow.pitch_name.isin([pitch]),"plate_z_norm"])
    ax.plot(prior,X,color=order_glasnow.loc[order_glasnow.pitch.isin([pitch]),"color"].tolist()[0],label="Prior")

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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Glasnow","Figure1-v3-PlatePositionByPitch-Glasnow.svg"))

#########################################
#                                       #
#             Make P(z|velo)            #
#                                       #
#########################################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

glasnow.dropna(subset=['release_speed'],inplace=True)
speed_bins = [glasnow.release_speed.min(),90,glasnow.release_speed.max()]
glasnow.loc[:,'release_speed_binned'] = pd.cut(glasnow['release_speed'].astype("float"),bins=speed_bins)

for cnt,(bin,clr,lnstyle) in enumerate(zip(glasnow.loc[:,'release_speed_binned'].unique(),['black','grey'],['-','--'])):
    # Get pdf for glasnow data - aka prior
    mean,std=stats.norm.fit(glasnow.loc[glasnow.release_speed_binned == bin,'plate_z_norm'])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std)
    prior /= len(glasnow.loc[glasnow.release_speed_binned == bin,'plate_z_norm'])
    ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} mph".format(speed_bins[cnt],speed_bins[cnt+1]))


### USE QBINS ###
# bin_labels = list(map(str, range(1,5))) #['1','2','3','4','5','6','7','8','9']
# velo_qbinned , velo_qbins = pd.qcut(glasnow['release_speed'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
# glasnow.loc[:,'release_speed_qbinned'] = velo_qbinned

# clrs = ['black','darkgrey','grey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(glasnow.loc[:,'release_speed_binned'].unique(),['black','grey'],['-',':'])):
# for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','darkgrey'],['-',':','--'])):
#     # Get pdf for glasnow data - aka prior
#     mean,std=stats.norm.fit(glasnow.loc[glasnow.release_speed_qbinned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     prior /= np.sum(prior)
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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Glasnow","Figure1-v3-PlatePositionByVelo-Glasnow.svg"))


#########################################
#                                       #
#              Make P(z|spin)           #
#                                       #
#########################################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

glasnow = glasnow.dropna(subset=['spin_axis'])

spin_cutbins = [0,50,150,250]
glasnow.loc[:,'spin_axis_binned'] = pd.cut(glasnow['spin_axis'].astype("float"),bins=spin_cutbins)

clrs = ['black','darkgrey','grey']
for cnt,(bin,clr,lnstyle) in enumerate(zip(glasnow.loc[:,'spin_axis_binned'].unique(),['black','grey','darkgrey'],['-','--',':'])):
    # Get pdf for glasnow data - aka prior
    if len(glasnow.loc[glasnow.spin_axis_binned == bin,'plate_z_norm']) > 50:
        mean,std=stats.norm.fit(glasnow.loc[glasnow.spin_axis_binned == bin,'plate_z_norm'])
        X= np.linspace(-.5, 1.5, 100)
        prior = stats.norm.pdf(X, mean, std)
        ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} degrees".format(bin.left,bin.right))


#### USE QBINS ###
# bin_labels = list(map(str, range(1,6))) #['1','2','3','4','5','6','7','8','9']
# spin_qbinned , spin_qbins = pd.qcut(glasnow['spin_axis'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
# glasnow.loc[:,'spin_axis_qbinned'] = spin_qbinned

# clrs = ['black','darkgrey','grey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(glasnow.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
# for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','lightgrey'],['-','--',':'])):
#     # Get pdf for glasnow data - aka prior
#     mean,std=stats.norm.fit(glasnow.loc[glasnow.spin_axis_qbinned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     prior /= np.sum(prior)
#     ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{}".format(spin_qbins[cnt],spin_qbins[cnt+1]))

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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Glasnow","Figure1-v3-PlatePositionBySpinAxis-Glasnow.svg"))



# #################
# #               #
# #   Spin Rate   #
# #               #
# #################

# # Make figure
# fig = plt.figure(figsize=(1,3.5))
# ax = fig.add_subplot()

# glasnow = glasnow.dropna(subset=['release_spin_rate'])

# bin_labels = list(map(str, range(1,4))) #['1','2','3','4','5','6','7','8','9']
# spinrate_qbinned , spinrate_qbins = pd.qcut(glasnow['release_spin_rate'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
# glasnow.loc[:,'release_spin_rate_qbinned'] = spinrate_qbinned

# clrs = ['black','darkgrey','grey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(glasnow.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
# for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','lightgrey'],['-','--',':'])):
#     # Get pdf for glasnow data - aka prior
#     mean,std=stats.norm.fit(glasnow.loc[glasnow.release_spin_rate_qbinned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     prior /= np.sum(prior)
#     ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} rpm".format(spinrate_qbins[cnt],spinrate_qbins[cnt+1]))

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

# fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Glasnow","Figure1-v3-PlatePositionBySpinRate-Glasnow.svg"))


#########################
#                       #
#   Vertical Movement   #
#                       #
#########################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

glasnow = glasnow.dropna(subset=['pfx_z_norm'])

pfxz_cutbins = [glasnow['pfx_z_norm'].min(), -0.4 ,0.1, glasnow['pfx_z_norm'].max()]
glasnow.loc[:,'pfx_z_binned'], retbins = pd.cut(glasnow['pfx_z_norm'].astype("float"),bins=pfxz_cutbins,retbins=True)

clrs = ['black','darkgrey']
for cnt,(bin,clr,lnstyle) in enumerate(zip(glasnow.loc[:,'pfx_z_binned'].unique(),['black','grey','darkgrey'],['-',':','--'])):
    # Get pdf for glasnow data - aka prior
    print(bin)
    if len(glasnow.loc[glasnow.pfx_z_binned == bin,'plate_z_norm']) > 20:
        mean,std=stats.norm.fit(glasnow.loc[glasnow.pfx_z_binned == bin,'plate_z_norm'])
        X= np.linspace(-.5, 1.5, 100)
        prior = stats.norm.pdf(X, mean, std) / len(glasnow.loc[glasnow.pfx_z_binned == bin,'plate_z_norm'])
        ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} %".format(round(bin.left,2),round(bin.right,2)))

# bin_labels = list(map(str, range(1,5))) #['1','2','3','4','5','6','7','8','9']
# pfxz_qbinned , pfxz_qbins = pd.qcut(glasnow['pfx_z_float'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
# glasnow.loc[:,'pfx_z_qbinned'] = pfxz_qbinned

# clrs = ['black','darkgrey','grey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(glasnow.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
# for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','lightgrey'],['-','--',':'])):
#     # Get pdf for glasnow data - aka prior
#     mean,std=stats.norm.fit(glasnow.loc[glasnow.pfx_z_qbinned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     prior /= np.sum(prior)
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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Glasnow","Figure1-v3-PlatePositionByVerticalMovement-Glasnow.svg"))



# #########################
# #                       #
# #   Vertical Movement   #
# #                       #
# #########################

# # Make figure
# fig = plt.figure(figsize=(1,3.5))
# ax = fig.add_subplot()

# glasnow.loc[:,"vertical_drop"] = glasnow.loc[:,"release_pos_z_float"] - glasnow.loc[:,"plate_z"]
# glasnow.dropna(subset=['vertical_drop'],inplace=True)

# # vdrop_cutbins = [glasnow['vertical_drop'].min(), 0 , glasnow['pfx_z_float'].max()]
# # glasnow.loc[:,'pfx_z_binned'] = pd.cut(glasnow['pfx_z_float'].astype("float"),bins=pfxz_cutbins)

# # clrs = ['black','darkgrey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(glasnow.loc[:,'pfx_z_binned'].unique(),['black','grey'],['-',':'])):
# #     # Get pdf for glasnow data - aka prior
# #     mean,std=stats.norm.fit(glasnow.loc[glasnow.pfx_z_binned == bin,'plate_z_norm'])
# #     X= np.linspace(-.5, 1.5, 100)
# #     prior = stats.norm.pdf(X, mean, std)
# #     ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} inches".format(pfxz_cutbins[cnt],pfxz_cutbins[cnt+1]))

# bin_labels = list(map(str, range(1,5))) #['1','2','3','4','5','6','7','8','9']
# vdrop_qbinned , vdrop_qbins = pd.qcut(glasnow['vertical_drop'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
# glasnow.loc[:,'vertical_drop_qbinned'] = vdrop_qbinned

# clrs = ['black','darkgrey','grey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(glasnow.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
# for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','lightgrey','darkgrey'],['-','--',':','-.'])):
#     # Get pdf for glasnow data - aka prior
#     mean,std=stats.norm.fit(glasnow.loc[glasnow.vertical_drop_qbinned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     prior /= np.sum(prior)
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

# fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Glasnow","Figure1-v3-PlatePositionByVerticalDrop-Glasnow.svg"))

#########################################
#                                       #
#      Make scatter of spin vs velo     #
#                                       #
#########################################

fig = plt.figure(figsize=(2,1))
ax = fig.add_subplot()
ss = sns.scatterplot(data=glasnow.loc[glasnow.pitch_name.isin(pitches_glasnow),:],y='release_speed_float', x='spin_axis',hue="pitch_name",hue_order=order_glasnow.pitch.tolist(),palette=order_glasnow.color.tolist(),legend=False,size=2,linewidth=.1)
ax.set_xlabel("Spin axis (deg)")
ax.set_ylabel("Release velocity (mph)")
ax.set_title("Inferred velocity from spin")
plt.rcParams.update({'font.sans-serif':'Arial','font.size':8})
ax.set_yticks([80,90,100])
ax.set_ylim([75,105])
fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Glasnow","Figure1-v3-SpinAxisVsVelocity-Glasnow.svg"))

#########################################
#                                       #
#      Make P(launch angle | pitch)     #
#                                       #
#########################################

fig = plt.figure(figsize=(3,2.5))
ax = fig.add_subplot()

plt.rcParams.update({'font.sans-serif':'Arial','font.size':10})

# For each bin, plot as jittered scatter plot
for (bin, bin_loc) in zip(sorted(glasnow.plate_z_norm_qbinned.unique()),which_bins_mean):
    _yvals = glasnow.loc[glasnow.pitch_name.isin(pitches_glasnow) & (glasnow.plate_z_norm_qbinned==bin),:].launch_angle
    xwithnoise = len(_yvals)*[bin_loc] + np.random.normal(0,.001,len(_yvals))
    ax.scatter(xwithnoise,_yvals,s=5,color="lightgray")

# For each pitch type
thisx = np.linspace(-.65,1.65, 100)
# each_pitch_ols = []
# for row in order_glasnow.itertuples():
#     if (len(glasnow.loc[glasnow.pitch_name.isin([row.pitch])]) >= 100):
#         ols_result = run_regression(glasnow.loc[glasnow.pitch_name.isin([row.pitch])].plate_z_norm,glasnow.loc[glasnow.pitch_name.isin([row.pitch])].launch_angle)
#         ax.plot(thisx,ols_result.params[0] + (ols_result.params[1]*thisx),linewidth=2,linestyle=":",color=row.color,label=row.pitch)

# Regression line for all pitches_glasnow
ols_result = run_regression(glasnow.loc[glasnow.pitch_name.isin(pitches_glasnow)].plate_z_norm,glasnow.loc[glasnow.pitch_name.isin(pitches_glasnow)].launch_angle)
ax.plot(thisx,ols_result.params[0] + (ols_result.params[1]*thisx),linewidth=2, color='black',label="All pitches_glasnow")

ols_result = run_regression(glasnow_alds.loc[glasnow.pitch_name.isin(pitches_glasnow)].plate_z_norm,glasnow_alds.loc[glasnow.pitch_name.isin(pitches_glasnow)].launch_angle)
ax.plot(thisx,ols_result.params[0] + (ols_result.params[1]*thisx),linewidth=2, color=order_greinke.loc[order_greinke.pitch.isin(["4-Seam Fastball"]),"color"].tolist()[0],label="All pitches_glasnow")

# Clean up
ax.set_yticks([-100,-50,0,50,100])
ax.set_xticks([0,1])
ax.set_xticklabels(["Knees","Chest"],color="firebrick",fontweight="bold")
# ax.spines['top'].set_visible(True)
# ax.spines['right'].set_visible(True)
ax.set_xlabel("Vertical Plate Position (% strike zone)",fontweight="bold")
ax.set_ylabel("Launch Angle (deg)",fontweight="bold")


fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Glasnow","Figure1-v3-LaunchAngleVersusPlateZ-Glasnow.svg"))

plt.rcParams.update({'font.sans-serif':'Arial','font.size':10})


#########################################
#                                       #
#           Make P(pitch)            #
#                                       #
#########################################

order_glasnow = order_glasnow.set_index("pitch").loc[glasnow.pitch_name.value_counts().index.tolist()]

# Make figure
fig = plt.figure(figsize=(2,2.5))
ax = fig.add_subplot()

countplot=sns.histplot(data=glasnow[glasnow.pitch_name.isin(pitches)],y="player_name",hue="pitch_name",hue_order=glasnow.pitch_name.value_counts(sort=True).index.tolist(),palette=order_glasnow.color.tolist(), multiple="dodge", 
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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Glasnow","Figure1-v3-PitchDistribution-Glasnow.svg"))


order_glasnow.reset_index(inplace=True)