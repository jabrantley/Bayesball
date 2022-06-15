########################################################################
#                                                                      #
#                MAKE FIGURE - TYLER GLASNOW PITCH TIPPING             #   
#                                                                      #
########################################################################

# Get data for Tyler Glasnow
tg_all  = data[data.game_year.isin([2018, 2019, 2020]) & data.player_name.isin(["Glasnow, Tyler"])]
tg_alds = data[data.game_date.isin(["2019-10-10"]) & data.player_name.isin(["Glasnow, Tyler"])]

###########################################
#                                         #
#   Define figure layout using gridspec   #
#                                         #
###########################################

# Define figure
fig = plt.figure(figsize=(7.5,2.85))
plt.rcParams.update({'font.sans-serif':'Arial','font.size':9})

# Add first gridspec for left and middle axes
gs0 = fig.add_gridspec(nrows=1, ncols=2, left=0.001, right=0.3,bottom=.01,top=0.95, wspace=0)
ax0 = fig.add_subplot(gs0[0])
ax00 = fig.add_subplot(gs0[1])

gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.425, right=0.99,top=.95, wspace=.35)
ax1 = fig.add_subplot(gs1[0])

# Add second gridspec for right axis
gs2 = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1,2], left=0.5, bottom=.65, top=.935, right=0.95,hspace=0.15)
ax2 = fig.add_subplot(gs2[0])
ax3 = fig.add_subplot(gs2[1])


###########################################
#                                         #
#      Add images for each pitch type     #
#                                         #
###########################################
# Add images
plt.sca(ax0)
img = mpimg.imread('images/tyler-glasnow-pitchtipping-Fastball.jpg')
ax0.imshow(img)
ax0.axis('off') 
sns.despine()
ax0.set_title("Fastball", fontsize=11, fontweight='bold',y=-0.085)

# Add images
plt.sca(ax00)
img = mpimg.imread('images/tyler-glasnow-pitchtipping-Curveball.jpg')
ax00.imshow(img)
ax00.axis('off') 
sns.despine()
ax00.set_title("Curveball", fontsize=11, fontweight='bold',y=-0.085)

# Make violin plot
plt.sca(ax1)

########################################################
#                                                      #
#    Make violinplot and add prior/likelihood lines    #
#                                                      #
########################################################

# Draw lines for prior and likelihood
likelihood_line = ax1.plot(extended_xvals,len(extended_xvals)*[results_by_pitch.loc[results_by_pitch.pitch=="All","error_at_mid"].iloc[0]],color="grey",linewidth=2.5,zorder=1,label="Mostly likelihood")
prior_bias = results_by_pitch.loc[results_by_pitch.pitch=="All","error_at_mid"].iloc[0] - (-0.075*0.5)
prior_line = ax1.plot(extended_xvals,(extended_xvals*-0.075 + prior_bias),linestyle=(0,(1,1)),color="black",linewidth=3,zorder=1,label="Mostly Prior")

# Make violinplot
my_violinplot(local_data=tg_all,
              true_shift=which_data_true_shift,
              deviation=which_data_deviation,
              ax=ax1,
              which_pitches=which_pitches,
              bin_locs=which_bins_mean)

# Compute regression  for each game
slope_vals = []
error_vals = []
game_ids   = []
for game_id in tg_all.game_pk.unique():
    try:
        tempdat = tg_all[(tg_all.game_pk.isin([game_id])) & (tg_all.pitch_name.isin(which_pitches))]
        xx = np.array(tempdat[which_data_true_shift])#_qbinned_percent_mean
        yy = np.array(tempdat[which_data_deviation])
        if (len(xx) >= 10):
            # Run regression for each game
            ols_result = run_regression(xx,yy)
            slope_vals.append(ols_result.params[1])
            error_vals.append(ols_result.params[0] + (ols_result.params[1] * (0)))
            
            # Get data from tipping game
            if (game_id == 599341):
                game_ids.append( "Tipping")
            else:
                game_ids.append("Rest of Season")
            # #     
            # xvals = np.unique(xx)
            # yvals = ols_result.params[0] + ols_result.params[1] * xvals

    except:
        print(len(xx),len(yy))

# Add data to dataframe
scatter_data = pd.DataFrame({"slope": slope_vals,"abs_slope": [abs(ii) for ii in slope_vals], "error": error_vals,"game_ids": game_ids})

# Regression line params
cmap_local = cm.get_cmap("plasma")
local_color = (0.1,0.1,0.1) #order.color[3]
regression_color = cmap_local(0.5)#order.color[5]

# Perform regression for all pitches vs alds tipping game
sns.regplot(data=tg_all,x=which_data_true_shift,y=which_data_deviation,scatter=False,ci=95,color=local_color,label="No Tipping")
sns.regplot(data=tg_alds,x=which_data_true_shift,y=which_data_deviation,scatter=False,ci=95,color=regression_color,label="Tipping")#,color="pink")

# Add labels
plt.gca().invert_xaxis()
ax1.set_xticks([0,0.5,1])
ax1.set_xticklabels(["Knees (0 %)", "50 %", "Chest (100 %)"])
ax1.get_xticklabels()[0].set_color("firebrick")
ax1.get_xticklabels()[2].set_color("firebrick")
ax1.get_xticklabels()[0].set_weight("bold")
ax1.get_xticklabels()[2].set_weight("bold")
ax1.set_xlabel("Vertical plate position (% strike zone)",fontweight='bold')
ax1.set_ylabel("Vertical contact error (% strike zone)",fontweight='bold')
ax1.set_title("Contact error vs Position in Strike Zone", fontsize=9, fontweight='bold',y=1)
ax1.set_ylim(-.05, .225)
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.legend(loc="upper left",frameon=True, bbox_to_anchor=(-.51,1.05),ncol=2)
    

###########################################
#                                         #
#  Make boxplot/swarmplot and histogram   #
#                                         #
###########################################

plt.sca(ax2)
sns.boxplot(x="abs_slope", data=scatter_data[scatter_data.game_ids=="Rest of Season"],color=local_color,boxprops=dict(alpha=.5))
sns.swarmplot(x="abs_slope", data=scatter_data[scatter_data.game_ids=="Rest of Season"],color=local_color)
sns.swarmplot(x="abs_slope", data=scatter_data[scatter_data.game_ids=="Tipping"],color=regression_color,s=20,marker="*",label="$\\bf{p < 0.001}$")
ax2leg = ax2.legend(loc="lower center",frameon=False,prop={"size":10}, markerscale=.75,bbox_to_anchor=(0.875,-.95))
ax2.axis('off')
ax2.set_zorder(3)

plt.sca(ax3)
sns.histplot(x="slope", data=scatter_data[scatter_data.game_ids=="Rest of Season"],bins=11,color=regression_color,kde=True)
ax3.set_ylabel(None)
ax3.set_xlabel(None)
ax3.yaxis.set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.set_title("Slope (per game)", fontsize=11, fontweight='bold',y=.45, x = 0.08)
plt.yticks(fontsize=9, rotation=70)

plt.show()

# Compute statistics on tipping vs no tipping
meanval, stdval = scatter_data[scatter_data.game_ids=="Rest of Season"].slope.describe()[['mean','std']]
n_samples = len(scatter_data.game_ids)-1
std_error = stdval/np.sqrt(n_samples)
t_value = (scatter_data[scatter_data.game_ids=="Tipping"].slope.to_numpy()[0] - meanval)/std_error
dof = len(scatter_data.game_ids)-2
t_critical = stats.t.ppf(q=.975,df=dof)

pval = stats.t.sf(np.abs(t_value), dof)*2 

print("T-value = ", t_value , "; T-critical = ", t_critical, "; P-val = ", pval)

fig.savefig(os.path.join(os.getcwd(),"figures","Figure4-pitchtipping.png"), dpi=300, facecolor='w', edgecolor='w', bbox_inches="tight")
fig.savefig(os.path.join(os.getcwd(),"figures","Figure4-pitchtipping.svg"))

plt.rcParams.update({'font.sans-serif':'Arial','font.size':10})
