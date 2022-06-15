########################################################################
#                                                                      #
#                    MAKE SUBPLOTS FOR FIGURE 1                        #
#                                                                      #
########################################################################

# The v2 figure 1 is made in inkscape, but the data figures are made here

# Pitches and order_greinke
pitches_greinke = ["Sinker","Cutter","4-Seam Fastball","Slider","Curveball","Changeup"]    
markers = ["X","o","*","s","^","P"]#["o","o"ax_top_L.spines['bottom'].set_color(tuple(3*[0.2])),"o","o","o","o"]#,"D"]#,"v"]    
cmap =  "colorblind"#"viridis"#
greinke_colors = []
for row in sns.color_palette(cmap,as_cmap=False,n_colors=len(pitches_greinke)): #frequent_pitches_greinke.index)):
    greinke_colors.append([*row,1.])    
order_greinke = pd.DataFrame({"pitch": [], "colormap": [], "color": [], "marker": []})
for (pitch,mrkr,this_clr) in zip(pitches_greinke,markers,greinke_colors):   
    order_greinke = pd.concat([order_greinke,pd.DataFrame({"pitch": pitch, "color": [this_clr],"marker": mrkr, "code": pitch_type_codes[pitch]})],ignore_index=True, axis=0)

# order_greinke = make_color_order_greinke(graded=False,cmap="viridis")
order_greinke.reset_index(inplace=True)

greinke = data.loc[(data.player_name.isin(["Greinke, Zack"])) & (data.pitch_name.isin(pitches_greinke)),:].copy()

# greinke["release_spin_rate"] = greinke["release_spin_rate"].fillna(0)
greinke.loc[:,"release_speed_float"] = greinke.release_speed.astype("float64")
greinke.loc[:,"release_spin_float"] = greinke.release_spin_rate.astype("float64")
greinke.loc[:,"release_pos_x_float"] = greinke.release_pos_x.astype("float64")
greinke.loc[:,"release_pos_z_float"] = greinke.release_pos_z.astype("float64")
greinke.loc[:,"pfx_x_float"] = greinke.pfx_x.astype("float64")
greinke.loc[:,"pfx_z_float"] = greinke.pfx_z.astype("float64")
greinke.loc[:,"plate_x_float"] = greinke.plate_x.astype("float64")
greinke.loc[:,"plate_z_float"] = greinke.plate_z.astype("float64")

greinke.loc[:,"plate_x_norm"] = np.array((greinke.plate_x - (-0.71)) / (0.71 - (-0.71)),dtype="float")
greinke.loc[:,"plate_z_norm"] = np.array((greinke.plate_z - greinke.sz_bot) / (greinke.sz_top - greinke.sz_bot), dtype="float")

#########################################
#                                       #
#       Make full bayes for batter      #
#                                       #
#########################################

# Get pdf for greinke data - aka prior
mean,std=stats.norm.fit(greinke.loc[greinke.pitch_name.isin(["4-Seam Fastball"]),"plate_z_norm"]) #mean,std=stats.norm.fit(greinke.loc().plate_z_norm)
X= np.linspace(-.5, 1.5, 100)
greinke_prior = stats.norm.pdf(X, mean, std)
greinke_prior /= np.sum(greinke_prior)

# Get likelihood
likelihood = stats.norm.pdf(X, 0.05, 0.2)
likelihood /= np.sum(likelihood)

# Get posterior
posterior = np.multiply(greinke_prior,likelihood)
posterior /= np.sum(posterior) 

# Make figure
fig = plt.figure(figsize=(1.5,4))
ax = fig.add_subplot()

ax.plot(greinke_prior,X,color="dodgerblue",zorder=2,label="Prior") #order_greinke.loc[order_greinke.pitch.isin(["4-Seam Fastball"]),"color"].tolist()[0]
# ax.fill_between(greinke_prior,X,color="dodgerblue",zorder=2,alpha=0.2)
ax.plot(likelihood,X,color=(0.5,0.5,0.5),zorder=3,label="Likelihood")
# ax.fill_between(likelihood,X,color=(0.5,0.5,0.5),zorder=3,alpha=0.2)
ax.plot(posterior,X,color=order_greinke[order_greinke.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,label="Posterior")
# ax.fill_between(posterior,X,color=order_greinke[order_greinke.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,alpha=0.2)
ax.scatter(posterior.max(),X[posterior.argmax()],s=50,color=order_greinke[order_greinke.pitch.isin(["4-Seam Fastball"])].color.tolist()[0],zorder=5,marker="*",label="Estimate")

# Make figure adjustments
ax.set_ylim(-.5,1.5)
bin_ticks=[-.5,0,0.25,0.5,0.75,1,1.5]
ax.set_yticks([0,1])#ticks=bin_ticks)
ax.set_yticklabels([])#list(map(str, [round(val,2) for val in bin_ticks])))#['1','2','3','4','5','6','7','8'])
ax.tick_params(axis='y')#,direction='out')
ax.set_ylabel(None)
ax.xaxis.set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Greinke","Figure1-v2-GreinkePriorLikelihoodPosterior.svg"))

#########################################
#                                       #
#           Make P(z| pitch)            #
#                                       #
#########################################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

for pitch in pitches_greinke:
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.pitch_name.isin([pitch]),"plate_z_norm"])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std)
    prior /= len(greinke.loc[greinke.pitch_name.isin([pitch]),:])
    ax.plot(prior,X,color=order_greinke.loc[order_greinke.pitch.isin([pitch]),"color"].tolist()[0],label="Prior")

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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Greinke","Figure1-v3-PlatePositionByPitch-Greinke.svg"))


#########################################
#                                       #
#             Make P(z|velo)            #
#                                       #
#########################################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

greinke.dropna(subset=['release_speed'],inplace=True)
cut_bins = [greinke.release_speed.min(),80,greinke.release_speed.max()]
greinke.loc[:,'release_speed_binned'] = pd.cut(greinke['release_speed'].astype("float"),bins=cut_bins)


# bin_labels = 4#list(map(str, range(1,4))) #['1','2','3','4','5','6','7','8','9']
# greinke.loc[:,'release_speed_binned'] = pd.cut(greinke['release_speed'].astype("float"),bins=bin_labels)

clrs = ['black','darkgrey','grey']
# for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'release_speed_binned'].unique(),['black','grey'],['-',':'])):
for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'release_speed_binned'].unique(),['black','grey'],['-','--'])):
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.release_speed_binned == bin,'plate_z_norm'])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std)
    prior /= len(greinke.loc[greinke.release_speed_binned == bin,'plate_z_norm'])
    ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} mph".format(cut_bins[cnt],cut_bins[cnt+1]))


# bin_labels = list(map(str, range(1,4))) #['1','2','3','4','5','6','7','8','9']
# velo_qbinned , velo_qbins = pd.qcut(greinke['release_speed'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
# greinke.loc[:,'release_speed_qbinned'] = velo_qbinned

# clrs = ['black','darkgrey','grey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'release_speed_binned'].unique(),['black','grey'],['-',':'])):
# for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','darkgrey'],['-',':','--'])):
#     # Get pdf for greinke data - aka prior
#     mean,std=stats.norm.fit(greinke.loc[greinke.release_speed_qbinned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     prior /= len(greinke.loc[greinke.release_speed_qbinned == bin,'plate_z_norm'])
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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Greinke","Figure1-v3-PlatePositionByVelo-Greinke.svg"))


#########################################
#                                       #
#              Make P(z|spin)           #
#                                       #
#########################################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

greinke.dropna(subset=['spin_axis'],inplace=True)

spin_cutbins = [0,150,greinke.spin_axis.max()]
greinke.loc[:,'spin_axis_binned'] = pd.cut(greinke['spin_axis'].astype("float"),bins=spin_cutbins)

clrs = ['black','darkgrey','grey']
for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.spin_axis_binned == bin,'plate_z_norm'])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std)
    prior /= len(greinke.loc[greinke.spin_axis_binned == bin,'plate_z_norm'])
    ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} degrees".format(spin_cutbins[cnt],spin_cutbins[cnt+1]))


# bin_labels = list(map(str, range(1,4))) #['1','2','3','4','5','6','7','8','9']
# spin_qbinned , spin_qbins = pd.qcut(greinke['spin_axis'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
# greinke.loc[:,'spin_axis_qbinned'] = spin_qbinned

# clrs = ['black','darkgrey','grey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
# for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','lightgrey'],['-','--',':'])):
#     # Get pdf for greinke data - aka prior
#     mean,std=stats.norm.fit(greinke.loc[greinke.spin_axis_qbinned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     prior /= len(greinke.loc[greinke.spin_axis_qbinned == bin,'plate_z_norm'])
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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Greinke","Figure1-v3-PlatePositionBySpinAxis-Greinke.svg"))


# #################
# #               #
# #   Spin Rate   #
# #               #
# #################

# # Make figure
# fig = plt.figure(figsize=(1,3.5))
# ax = fig.add_subplot()

# greinke.dropna(subset=['release_spin_rate'],inplace=True)

# bin_labels = list(map(str, range(1,4))) #['1','2','3','4','5','6','7','8','9']
# spinrate_qbinned , spinrate_qbins = pd.qcut(greinke['release_spin_rate'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
# greinke.loc[:,'release_spin_rate_qbinned'] = spinrate_qbinned

# clrs = ['black','darkgrey','grey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
# for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','lightgrey'],['-','--',':'])):
#     # Get pdf for greinke data - aka prior
#     mean,std=stats.norm.fit(greinke.loc[greinke.release_spin_rate_qbinned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     prior /= len(greinke.loc[greinke.release_spin_rate_qbinned == bin,'plate_z_norm'])
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

# fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Greinke","Figure1-v3-PlatePositionBySpinRate-Greinke.svg"))


# #########################
# #                       #
# #   Vertical Movement   #
# #                       #
# #########################

# # Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

greinke.dropna(subset=['pfx_z_norm'],inplace=True)

pfxz_cutbins = [-1,-0.25,0.5,1.5]#[greinke['pfx_z_float'].min(), 0 , greinke['pfx_z_float'].max()]
greinke.loc[:,'pfx_z_binned'] = pd.cut(greinke['pfx_z_norm'].astype("float"),bins=pfxz_cutbins)

clrs = ['black','darkgrey','lightgrey']
for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'pfx_z_binned'].unique(),clrs,['-',':','--'])):
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.pfx_z_binned == bin,'plate_z_norm'])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std) / len(greinke.loc[greinke.pfx_z_binned == bin,'plate_z_norm'])
    ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} inches".format(pfxz_cutbins[cnt],pfxz_cutbins[cnt+1]))
    
# bin_labels = list(map(str, range(1,4))) #['1','2','3','4','5','6','7','8','9']
# pfxz_qbinned , pfxz_qbins = pd.qcut(greinke['pfx_z_float'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
# greinke.loc[:,'pfx_z_qbinned'] = pfxz_qbinned

# clrs = ['black','darkgrey','grey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
# for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','lightgrey'],['-','--',':'])):
#     # Get pdf for greinke data - aka prior
#     mean,std=stats.norm.fit(greinke.loc[greinke.pfx_z_qbinned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     prior /= len(greinke.loc[greinke.pfx_z_qbinned == bin,'plate_z_norm'])
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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Greinke","Figure1-v3-PlatePositionByVerticalMovement-Greinke.svg"))



# #########################
# #                       #
# #   Vertical Movement   #
# #                       #
# #########################

# # Make figure
# fig = plt.figure(figsize=(1,3.5))
# ax = fig.add_subplot()

# greinke.loc[:,"vertical_drop"] = greinke.loc[:,"release_pos_z_float"] - greinke.loc[:,"plate_z"]
# greinke.dropna(subset=['vertical_drop'],inplace=True)

# # vdrop_cutbins = [greinke['vertical_drop'].min(), 0 , greinke['pfx_z_float'].max()]
# # greinke.loc[:,'pfx_z_binned'] = pd.cut(greinke['pfx_z_float'].astype("float"),bins=pfxz_cutbins)

# # clrs = ['black','darkgrey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'pfx_z_binned'].unique(),['black','grey'],['-',':'])):
# #     # Get pdf for greinke data - aka prior
# #     mean,std=stats.norm.fit(greinke.loc[greinke.pfx_z_binned == bin,'plate_z_norm'])
# #     X= np.linspace(-.5, 1.5, 100)
# #     prior = stats.norm.pdf(X, mean, std)
# #     ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} inches".format(pfxz_cutbins[cnt],pfxz_cutbins[cnt+1]))

# bin_labels = list(map(str, range(1,5))) #['1','2','3','4','5','6','7','8','9']
# vdrop_qbinned , vdrop_qbins = pd.qcut(greinke['vertical_drop'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
# greinke.loc[:,'vertical_drop_qbinned'] = vdrop_qbinned

# clrs = ['black','darkgrey','grey']
# # for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
# for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','lightgrey','darkgrey'],['-','--',':','-.'])):
#     # Get pdf for greinke data - aka prior
#     mean,std=stats.norm.fit(greinke.loc[greinke.vertical_drop_qbinned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     prior /= len(greinke.loc[greinke.vertical_drop_qbinned == bin,'plate_z_norm'])
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

# fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Greinke","Figure1-v3-PlatePositionByVerticalDrop-Greinke.svg"))

#########################################
#                                       #
#      Make scatter of spin vs velo     #
#                                       #
#########################################

fig = plt.figure(figsize=(2,1))
ax = fig.add_subplot()
ss = sns.scatterplot(data=greinke.loc[greinke.pitch_name.isin(pitches_greinke),:],y='release_speed_float', x='spin_axis',hue="pitch_name",hue_order=order_greinke.pitch.tolist(),palette=order_greinke.color.tolist(),legend=False,size=2,linewidth=.1)
ax.set_xlabel("Spin axis (deg)")
ax.set_ylabel("Release velocity (mph)")
ax.set_title("Inferred velocity from spin")
plt.rcParams.update({'font.sans-serif':'Arial','font.size':8})
ax.set_yticks([60,70,80,90])
ax.set_ylim([55,100])
fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Greinke","Figure1-v3-SpinAxisVsVelocity-Greinke.svg"))

#########################################
#                                       #
#      Make P(launch angle | pitch)     #
#                                       #
#########################################

# Make figure
fig = plt.figure(figsize=(1.5,3.5))
ax = fig.add_subplot()

for pitch in pitches_greinke:
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.pitch_name.isin([pitch]),"launch_angle"])
    X= np.linspace(-100, 100, 100)
    _P = stats.norm.pdf(X, mean, std)
    _P /= len(greinke.loc[greinke.pitch_name.isin([pitch]),"launch_angle"])
    ax.plot(_P,X,color=order_greinke.loc[order_greinke.pitch.isin([pitch]),"color"].tolist()[0])

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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Greinke","Figure1-v3-LaunchAngleByPitch-Greinke.svg"))


#########################################
#                                       #
#        Make P(spin axis| pitch)       #
#                                       #
#########################################

# Make figure
fig = plt.figure(figsize=(1.5,0.75))
ax = fig.add_subplot()

greinke.loc[:,"spin_axis"] = greinke.loc[:,"spin_axis"].astype("float")

for pitch in pitches_greinke:
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.pitch_name.isin([pitch]),"spin_axis"])
    X= np.linspace(0, 360, 360)
    prior = stats.norm.pdf(X, mean, std)
    prior /= len(greinke.loc[greinke.pitch_name.isin([pitch]),:])
    ax.plot(X,prior,color=order_greinke.loc[order_greinke.pitch.isin([pitch]),"color"].tolist()[0],label=pitch)

# Make figure adjustments
# ax.set_xlim(-.5,1.5)
ax.set_xticks([0,90,180,270,360])
# ax.set_xticklabels(["Pure Topspin","Sidespin \n" + r"(ball moves L $\rightarrow$ R)","Pure Backspin","Sidespin \n" + r"(ball moves R $\rightarrow$ L)",""])
ax.set_xticklabels(["Topspin","Sidespin","Backspin","Sidespin","Topspin"],fontsize=5)
ax.tick_params(axis='x',rotation=10)
ax.set_xlabel(None)
ax.yaxis.set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Greinke","Figure1-v3-SpinAxisByPitch-Greinke.svg"))



##########################################################
#                                                        #
#      Make scatter of launch angle vs plate position    #
#                                                        #
##########################################################

# Make violin instead
# my_violinplot(local_data=greinke,
#               true_shift="plate_z_norm",
#               deviation="launch_angle",
#               ax=ax,
#               which_pitches_greinke=pitches_greinke,
#               bin_locs=which_bins_mean)

fig = plt.figure(figsize=(2,1))
ax = fig.add_subplot()

# For each bin, plot as jittered scatter plot
for (bin, bin_loc) in zip(sorted(greinke.plate_z_norm_qbinned.unique()),which_bins_mean):
    _yvals = greinke.loc[greinke.pitch_name.isin(pitches_greinke) & (greinke.plate_z_norm_qbinned==bin),:].launch_angle
    xwithnoise = len(_yvals)*[bin_loc] + np.random.normal(0,.001,len(_yvals))
    ax.scatter(xwithnoise,_yvals,s=5,color="lightgray")

# For each pitch type
# thisx = np.linspace(-.65,1.65, 100)
# each_pitch_ols = []
# for row in order_greinke.itertuples():
#     if (len(greinke.loc[greinke.pitch_name.isin([row.pitch])]) >= 100):
#         ols_result = run_regression(greinke.loc[greinke.pitch_name.isin([row.pitch])].plate_z_norm,greinke.loc[greinke.pitch_name.isin([row.pitch])].launch_angle)
#         ax.plot(thisx,ols_result.params[0] + (ols_result.params[1]*thisx),linewidth=2,linestyle=":",color=row.color,label=row.pitch)

# Regression line for all pitches_greinke
ols_result = run_regression(greinke.loc[greinke.pitch_name.isin(pitches_greinke)].plate_z_norm,greinke.loc[greinke.pitch_name.isin(pitches_greinke)].launch_angle)
ax.plot(thisx,ols_result.params[0] + (ols_result.params[1]*thisx),linewidth=2, color=MAPcolor,label="All pitches_greinke")

# Clean up
ax.set_yticks([-100,-50,0,50,100])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.rcParams.update({'font.sans-serif':'Arial','font.size':6})
ax.set_xlabel("Vertical Plate Position (% strike zone)",fontweight="bold",fontsize=6)
ax.set_ylabel("Launch Angle (deg)",fontweight="bold",fontsize=6)


fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V3","Greinke","Figure1-v3-LaunchAngleVersusPlateZ-Greinke.svg"))

plt.rcParams.update({'font.sans-serif':'Arial','font.size':10})