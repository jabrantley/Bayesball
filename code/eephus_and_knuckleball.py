########################################################################
#                                                                      #
#      MAKE FIGURE - SIDE BY SIDE KNUCKLEBALL AND EEPHUS               #   
#                                                                      #
########################################################################

# Define figure
fig = plt.figure(figsize=(7,4))

# Add first gridspec for violinplot
gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.45, bottom=0.05,top=0.9)
ax1 = fig.add_subplot(gs1[0])

# Add gridspec for slope bar plot
gs2 = fig.add_gridspec(nrows=2, ncols=1, left=0.55, right=0.99,bottom=.05,top=.9,hspace=.3)
ax2 = fig.add_subplot(gs2[0])
ax3 = fig.add_subplot(gs2[1])

# gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.95, bottom=0.35,top=0.95)
# ax1 = fig.add_subplot(gs1[0])

#### Define colors ####
cmap_all = "viridis"
# all_clr = cmap_all(0.5)
cmap = cm.get_cmap('plasma')
# define_colors = [cmap(0.8), cmap(0.6),cmap(0.4), cmap(0.2)]
# define_colors = [colors.to_rgba("firebrick"), colors.to_rgba("mediumseagreen")]#[cmap(0.8), cmap(0.5)]#, cmap(0.2)]
define_colors = [cmap(0.8), cmap(0.5)]#, cmap(0.2)]

these_colors = []
for row in sns.color_palette(cmap_all,as_cmap=False,n_colors=len(which_pitches)): #frequent_pitches.index)):
    these_colors.append([*row,1.])
markers = ["X","^","*","s","o","P","D"]

##################################################
#                                                #   
#        Make violin plot for Knuckleball        #
#                                                #
##################################################

plt.sca(ax1)

likelihood_line = ax1.plot(extended_xvals,len(extended_xvals)*[results_by_pitch.loc[results_by_pitch.pitch=="All","error_at_mid"].iloc[0]],color="grey",linewidth=2.5,zorder=1,label="Mostly likelihood")
prior_bias = results_by_pitch.loc[results_by_pitch.pitch=="All","error_at_mid"].iloc[0] - (-0.075*0.5)
prior_line = ax1.plot(extended_xvals,(extended_xvals*-0.075 + prior_bias),linestyle=(0,(1,1)),color="black",linewidth=3,zorder=1,label="Mostly Prior")

my_violinplot(local_data=data,
              true_shift=which_data_true_shift,
              deviation=which_data_deviation,
              ax=ax1,
              which_pitches=which_pitches,
              bin_locs=which_bins_mean)

# Get data for plotting
xvals = np.unique(xx_all)
yvals = results_by_pitch[results_by_pitch.pitch.isin(["All"])].const.values[0] + (results_by_pitch[results_by_pitch.pitch.isin(["All"])].means.values[0] * xvals)

# Plot regression line
regression_color = 'black'#'dodgerblue' #np.array((0, 158, 115))/256) # blue: 0,114,178 ; green = 0, 158, 115 ; pink = 204,121,167
reg = ax1.plot(xvals,yvals,lw=2,color=regression_color,label="All")


knuckleballs = data[data.pitch_name.isin(['Knuckleball'])]
sns.regplot(data=knuckleballs,x=which_data_true_shift,y=which_data_deviation,scatter=False,ci=95,label="Knuckleball",line_kws=dict(lw=2),color=define_colors[0])
        
eephus = data[data.pitch_name.isin(['Eephus'])]#data[data.release_speed < 60] #
sns.regplot(data=eephus,x=which_data_true_shift,y=which_data_deviation,scatter=False,ci=95,label="Eephus",line_kws=dict(lw=2),color=define_colors[1])        
    
xx = np.array(knuckleballs[which_data_true_shift])#_qbinned_percent_mean
yy = np.array(knuckleballs[which_data_deviation])
X = sm.add_constant(xx)#, prepend=False)
ols = sm.OLS(yy,X)
ols_result = ols.fit()
slope_vals.append(ols_result.params[1])
error_vals.append(ols_result.params[0] + (ols_result.params[1] * (0.4)))
prstd_ols, iv_l_ols, iv_u_ols = wls_prediction_std(ols_result) # for getting confidence intervals
xvals = np.unique(xx)
yvals = ols_result.params[0] + ols_result.params[1] * xvals

            
knuckleball_df = pd.DataFrame({'pitch': ['Knuckleball'],'pitch_code': ['KN'],
                               'const': [ols_result.params[0]],
                               'means': [ols_result.params[1]], 
                               'absmeans': [abs(ols_result.params[1])], 
                               'se_mean': [[ols_result.bse[0], ols_result.bse[1]]],
                               'se_mean_lo': [ols_result.params[1] - ols_result.bse[0]],
                               'se_mean_hi': [ols_result.params[1] + ols_result.bse[1]],
                                'error_at_mid': [ols_result.params[0] + ols_result.params[1] * (0)]})

knuckleball_results = results_by_pitch.sort_values("absmeans",ascending=False).append(knuckleball_df)
    

xx = np.array(eephus[which_data_true_shift])#_qbinned_percent_mean
yy = np.array(eephus[which_data_deviation])
X = sm.add_constant(xx)#, prepend=False)
ols = sm.OLS(yy,X)
ols_result = ols.fit()
slope_vals.append(ols_result.params[1])
error_vals.append(ols_result.params[0] + (ols_result.params[1] * (0.4)))
prstd_ols, iv_l_ols, iv_u_ols = wls_prediction_std(ols_result) # for getting confidence intervals
xvals = np.unique(xx)
yvals = ols_result.params[0] + ols_result.params[1] * xvals

            
eephus_df = pd.DataFrame({'pitch': ['Eephus'],'pitch_code': ['EP'],
                               'const': [ols_result.params[0]],
                               'means': [ols_result.params[1]], 
                               'absmeans': [abs(ols_result.params[1])], 
                               'se_mean': [[ols_result.bse[0], ols_result.bse[1]]],
                               'se_mean_lo': [ols_result.params[1] - ols_result.bse[0]],
                               'se_mean_hi': [ols_result.params[1] + ols_result.bse[1]],
                                'error_at_mid': [ols_result.params[0] + ols_result.params[1] * (0)]})

eephus_results = knuckleball_results.append(eephus_df)
eephus_results = eephus_results[eephus_results.pitch != "All"]

# Add labels
# ax1.set_xticks([0,0.5,1])
ax1.set_xticklabels(["Knees (0 %)", "50 %", "Chest (100 %)"])
ax1.get_xticklabels()[0].set_color("firebrick")
ax1.get_xticklabels()[2].set_color("firebrick")
ax1.get_xticklabels()[0].set_weight("bold")
ax1.get_xticklabels()[2].set_weight("bold")
ax1.set_xlabel("Vertical plate position (% strike zone)",fontweight='bold')
ax1.set_ylabel("Vertical contact error (% strike zone)",fontweight='bold')
ax1.set_title("Contact Error vs Position in Strike Zone", fontsize=11, fontweight='bold')
ax1.set_ylim(-.06, .175)
ax1.set_xlim(extended_xvals[0],extended_xvals[-1:])
ax1.set_xticks([0,0.5,1])
ax1.invert_xaxis()
ax1.legend(loc="upper left",frameon=False,fontsize=9,ncol=2) #, bbox_to_anchor=(-.4,1.5)
ax1.spines['top'].set_visible(True)    
ax1.spines['right'].set_visible(True)    

plt.sca(ax2)
eephus_results.plot(y="absmeans", x="pitch", kind="bar", yerr=np.array(eephus_results.se_mean.tolist()).T,legend=None,ax=ax2,color=list(these_colors + define_colors))#7*[(0.5,0.5,0.5,1)])# define_colors[:-1] + define_colors[:2] + [define_colors[-1]],)#plt.cm.Paired(np.arange(len(frequent_pitches.index))))#,colormap=these_colors)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.xaxis.set_visible(False)
ax2.set_xlabel(None)
ax2.set_ylabel("Slope",fontweight="bold")
ax2.set_yticks([0.0, 0.01, 0.02])
ax2.set_ylim(0,0.025)

plt.sca(ax3)
eephus_results.plot(y="const", x="pitch", kind="bar", yerr=np.array(eephus_results.se_mean.tolist()).T,legend=None,ax=ax3,color=list(these_colors + define_colors))#7*[(0.5,0.5,0.5,1)])# define_colors[:-1] + define_colors[:2] + [define_colors[-1]],)#plt.cm.Paired(np.arange(len(frequent_pitches.index))))#,colormap=these_colors)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.set_yticks([0.0, 0.01, 0.02])
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax3.set_xlabel(None)
ax3.set_ylabel("Bias",fontweight="bold")
ax3.tick_params(axis="x",length=0)
ax3.set_ylim(0,0.025)

ax1.text(1.90,.175,"A.",fontweight="bold",fontsize=16)
ax1.text(-.75,.175,"B.",fontweight="bold",fontsize=16)
ax1.text(-.75,0.04,"C.",fontweight="bold",fontsize=16)

# plt.tight_layout()
fig.savefig(os.path.join(os.getcwd(),"figures","Figure3-KnuckleballAndEephus.png"), dpi=300, facecolor='w', edgecolor='w',bbox_inches="tight")
