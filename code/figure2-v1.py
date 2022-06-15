########################################################################
#                                                                      #
#                          MAKE FIGURE AA - V2                         #
#                                                                      #
########################################################################

# Define colors
# from cv2 import rotate


regression_color = "dodgerblue"

cmap =  "viridis"#"plasma"#"crest"#"colorblind"
these_colors = []
for row in sns.color_palette(cmap,as_cmap=False,n_colors=len(which_pitches)): #frequent_pitches.index)):
    these_colors.append([*row,1.])
    
markers = ["X","^","*","s","o","P","D"]#,"v"]

# Get bin tick values based on .qbin()
bin_ticks = []
for ii in range(len(which_bins)):
    bin_ticks.append(np.mean(which_bins[ii:(ii+1)]))
bin_ticks_labels = list(map(str, [round(val,1) for val in bin_ticks]))
bin_ticks.extend([0,1])
bin_ticks_labels.extend(['0','1'])

# Define figures using gridspec
fig = plt.figure(constrained_layout=False)
fig.set_size_inches(7.5,8)
plt.rcParams.update({'font.sans-serif':'Arial','font.size':10})

###########################################
#                                         #
#   Define figure layout using gridspec   #
#                                         #
###########################################

# Add first gridspec for left and middle axes
gs1 = fig.add_gridspec(nrows=1, ncols=2, left=-0.001, right=0.325, wspace=.55,width_ratios=[1,.35],top=.99,bottom=0.55)
ax1 = fig.add_subplot(gs1[0, 0])
ax2 = fig.add_subplot(gs1[0, 1])

# Add second gridspec for right axis
gs2 = fig.add_gridspec(nrows=1, ncols=1, left=0.5, right=0.999,top=0.985,bottom=0.55)#
ax3 = fig.add_subplot(gs2[0,0]) 

# gs3 = fig.add_gridspec(nrows=1, ncols=1, left=0.58, right=0.985,bottom=0.8)
# ax4 = fig.add_subplot(gs3[0,0])

gs4 = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.35,bottom=0.025,top=0.425)
ax5 = fig.add_subplot(gs4[0,0])

gs5 = fig.add_gridspec(nrows=1, ncols=1, left=0.5, right=0.99,bottom=0.3,top=0.45)
ax6 = fig.add_subplot(gs5[0,0])

gs6 = fig.add_gridspec(nrows=1, ncols=1, left=0.5, right=0.99,bottom=0.05,top=0.225)
ax7 = fig.add_subplot(gs6[0,0])

##############################
#                            #       
#        Axis 1 [0]          #
#                            #
##############################

# Show batter image
img = mpimg.imread('images/batter.png')
ax1.imshow(img)
ax1.axis('off') 
sns.despine()
ax1.set_xlim(120,1100)
ax1.set_ylim(2500,90)

##############################
#                            #       
#        Axis 2 [1]          #
#                            #
##############################
plot_kde_only = False

if plot_kde_only:
    sns.kdeplot(data=data, y="plate_z_norm",ax=ax2,fill=False,color="k")
    xpos, ypos_chest, ypos_knees = .75, .9875, -.02
else:
    sns.histplot(data,y="plate_z_norm",bins=35,ax=ax2,color='silver',stat='density',kde=True,shrink=0.9,edgecolor="w", linewidth=1.5,zorder=2)
    sns.kdeplot(data=data[data.pitch_name.isin(which_pitches)], y="plate_z_norm",ax=ax2,fill=False,color="dodgerblue",zorder=2)#color="darkgrey")
    xpos, ypos_chest, ypos_knees = 1, 1.05, -.1

# Add lines to label strikezone
ax2.axhline(y = 1, xmin=-.01, xmax=0.85,color='firebrick',linestyle=':',clip_on=False,zorder=2,lw=2)
ax2.axhline(y = 0, xmin=-.01, xmax=0.85,color='firebrick',linestyle=':',clip_on=False,zorder=2,lw=2)
txt1 = ax2.text(xpos,ypos_chest,"Chest",color='firebrick',fontweight="bold",zorder=2)
txt2 = ax2.text(xpos,ypos_knees,"Knees",color='firebrick',fontweight="bold",zorder=2)

# Make figure adjustments
ax2.set_xlim(0,1.25)
ax2.set_ylim(-.75,1.6)
ax2.set_yticks(ticks=[0,.5,1])
# ax2.set_yticklabels(['0','50','100'])#bin_ticks_labels)#['1','2','3','4','5','6','7','8'])
# ax2.tick_params(axis='y')#,direction='out')
ax2.xaxis.set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.set_ylabel(None)
ax2.text(-1.1,0.45,"% Strikezone",fontweight='bold',va="center",rotation="vertical")
ax2.text(0,-1,"Home plate",fontsize=10,fontweight="bold",ha="center")
ax2.set_title("Vertical Pitch Position \n In Strike Zone", fontsize=10, fontweight='bold',y=1.02,x=-1)

##############################
#                            #       
#        Axis 3 [2]          #
#                            #
##############################

plt.sca(ax3)
xvals = np.unique(xx_all)
yvals = results_by_pitch[results_by_pitch.pitch.isin(["All"])].const.values[0] + (results_by_pitch[results_by_pitch.pitch.isin(["All"])].means.values[0] * xvals)
extended_xvals = np.array([-.5,*xvals,1.5])
extended_yvals = results_by_pitch[results_by_pitch.pitch.isin(["All"])].const.values[0] + (results_by_pitch[results_by_pitch.pitch.isin(["All"])].means.values[0] * extended_xvals)

# Draw lines showing slope=0 and slope~=1
likelihood_line = ax3.plot(extended_xvals,len(extended_xvals)*[results_by_pitch.loc[results_by_pitch.pitch=="All","error_at_mid"].iloc[0]],color="grey",linewidth=2.5,zorder=1,label="Mostly likelihood")
prior_bias = results_by_pitch.loc[results_by_pitch.pitch=="All","error_at_mid"].iloc[0] - (-0.075*0.5)
prior_line = ax3.plot(extended_xvals,(extended_xvals*-0.075 + prior_bias),linestyle=(0,(1,1)),color="black",linewidth=3,zorder=1,label="Mostly Prior")

# Make violin
my_violinplot(local_data=data,
              true_shift=which_data_true_shift,
              deviation=which_data_deviation,
              ax=ax3,
              which_pitches=which_pitches,
              bin_locs=which_bins_mean)

# Plot regression line #np.array((0, 158, 115))/256) # blue: 0,114,178 ; green = 0, 158, 115 ; pink = 204,121,167
reg = ax3.plot(extended_xvals,extended_yvals,lw=2.5,color=regression_color,label="All Pitches")#,zorder=3)

# Add labels
ax3.set_xlim([-.7,1.7])
plt.gca().invert_xaxis()
ax3.set_xticks([0,0.5,1])
ax3.set_ylim([-.075,0.2])
ax3.set_xticklabels(["Knees (0 %)", "50 %", "Chest (100 %)"])
ax3.get_xticklabels()[0].set_color("firebrick")
ax3.get_xticklabels()[2].set_color("firebrick")
ax3.get_xticklabels()[0].set_weight("bold")
ax3.get_xticklabels()[2].set_weight("bold")
ax3.set_xlabel("Vertical plate position (% of strike zone)",fontweight='bold')
ax3.set_ylabel("Vertical contact error (% of strike zone)",fontweight='bold',y=-.05)
ax3.set_title("Contact error vs Position in Strike Zone", fontsize=11, fontweight='bold')#,y=1.05)
ax3.yaxis.set_label_coords(-.175,.5)
ax3.spines['top'].set_visible(True)
ax3.spines['right'].set_visible(True)
ax3.legend(frameon=False)

##############################
#                            #       
#          Axis 4            #
#                            #
##############################

# Define colors and markers
regression_color = "dodgerblue"

cmap =  "viridis"#"plasma"#"crest"#"colorblind"
these_colors = []
for row in sns.color_palette(cmap,as_cmap=False,n_colors=len(which_pitches)): #frequent_pitches.index)):
    these_colors.append([*row,1.])
    
markers = ["X","^","*","s","o","P","D"]#,"v"]
    
# Make bottom left panel 
plt.sca(ax5)    
sns.kdeplot(data=data_by_pitcher_trim,x="ols_slope_abs",y="ols_const",color="lightgrey",ax=ax5,fill=False,levels=5)
sns.regplot(data=data_by_pitcher_trim,x="ols_slope_abs",y="ols_const",color="dodgerblue",ax=ax5,scatter=False)
scat = sns.scatterplot(data=results_by_pitch[results_by_pitch.pitch.isin(which_pitches)],x="absmeans",y="const",hue="pitch",style="pitch",s=100,ax=ax5,palette=cmap,hue_order=which_pitches,legend=False,markers=markers,zorder=3)#palette="crest")
ax5.spines['top'].set_visible(True)
ax5.spines['right'].set_visible(True)
ax5.set_xlabel("Bias",fontweight='bold')
ax5.set_ylabel("Slope",fontweight='bold')
ax5.set_title("Slope vs Bias Across All Pitch Types", fontsize=11, fontweight='bold')#,y=1.05)

# sort_order = data_by_pitcher_trim.groupby(by="pitch_name")["ols_slope_abs"].agg(['mean']).sort_values(by='mean',ascending=False).index.tolist()
sort_order = results_by_pitch[results_by_pitch.pitch.isin(which_pitches)].sort_values("absmeans",ascending=False).pitch.tolist()

plt.sca(ax6) 
results_by_pitch[results_by_pitch.pitch.isin(which_pitches)].sort_values("absmeans",ascending=False).plot(y="absmeans", x="pitch", kind="bar",color=these_colors, yerr=np.array(results_by_pitch[results_by_pitch.pitch.isin(which_pitches)].se_mean.tolist()).T,legend=None,ax=ax6)
ax6.spines['right'].set_visible(False)
ax6.spines['bottom'].set_visible(False)
ax6.xaxis.set_visible(False)
ax6.set_xlabel(None)
ax6.set_ylabel("Slope",fontweight="bold")
ax6.set_yticks([0.0, 0.01, 0.02])
ax6.set_ylim(0,0.025)

# Run stats
fvalue, pvalue = stats.f_oneway(data_by_pitcher[data_by_pitcher.pitch_name.isin(['4-Seam Fastball'])].ols_slope_abs,
                                      data_by_pitcher[data_by_pitcher.pitch_name.isin(['Slider'])].ols_slope_abs,
                                      data_by_pitcher[data_by_pitcher.pitch_name.isin(['Sinker'])].ols_slope_abs,
                                      data_by_pitcher[data_by_pitcher.pitch_name.isin(['Changeup'])].ols_slope_abs,
                                      data_by_pitcher[data_by_pitcher.pitch_name.isin(['2-Seam Fastball'])].ols_slope_abs,
                                      data_by_pitcher[data_by_pitcher.pitch_name.isin(['Curveball'])].ols_slope_abs,
                                      data_by_pitcher[data_by_pitcher.pitch_name.isin(['Cutter'])].ols_slope_abs,
                                      data_by_pitcher[data_by_pitcher.pitch_name.isin(['Split-Finger'])].ols_slope_abs)
print(pvalue)                                                                                                          

# Plot slope and bias bar plots
plt.sca(ax7)
results_by_pitch[results_by_pitch.pitch.isin(which_pitches)].sort_values("absmeans",ascending=False).plot(y="const", x="pitch", kind="bar", color=these_colors, yerr=np.array(results_by_pitch[results_by_pitch.pitch.isin(which_pitches)].se_mean.tolist()).T,legend=None,ax=ax7)
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
ax7.spines['bottom'].set_visible(False)
ax7.set_yticks([0.0, 0.01, 0.02])
plt.setp(ax7.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax7.set_xlabel(None)
ax7.set_ylabel("Bias",fontweight="bold")
for cnt,marker in enumerate(markers):
    ax7.scatter(x=cnt,y=-0.005,s=100,marker=marker,color=these_colors[cnt])
ax7.tick_params(axis="x",length=0)
ax7.set_ylim(-.01,0.025)

# Add patch to cover bottom 
from matplotlib.patches import Rectangle
ax7.add_patch(Rectangle((-0.55,-0.001),width=0.25,height=-0.04, zorder=3,facecolor="white"))

# Add labels
ax2.text(-5.5,1.75,"A.",fontweight="bold",fontsize=16)
ax2.text(3.,1.75,"B.",fontweight="bold",fontsize=16)
ax2.text(-5.5,-1.25,"C.",fontweight="bold",fontsize=16)
ax2.text(3.,-1.45,"D.",fontweight="bold",fontsize=16)
ax2.text(3.,-3.15,"E.",fontweight="bold",fontsize=16)
ax2.plot((-1,1),(-0.8,-0.8),'k-',linewidth=2.5,clip_on=False)
ax2.add_patch(Rectangle((-1,0),width=2,height=1, zorder=1,fill=False,facecolor=None,edgecolor="lightgrey",clip_on=False))

# Save figure
plt.show()
fig.savefig(os.path.join(os.getcwd(),"figures","Figure2-moneyplot_withSlopeandBias.png"), dpi=300, facecolor='w', edgecolor='w', bbox_inches="tight")
