########################################################################
#                                                                      #
#                    MAKE SUBPLOTS FOR FIGURE 1                        #
#                                                                      #
########################################################################

# The v2 figure 1 is made in inkscape, but the data figures are made here

# Pitches and order
pitches = ["Sinker","Cutter","4-Seam Fastball","Slider","Curveball","Changeup"]    
markers = ["X","o","*","s","^","P"]#["o","o"ax_top_L.spines['bottom'].set_color(tuple(3*[0.2])),"o","o","o","o"]#,"D"]#,"v"]    
cmap =  "colorblind"#"viridis"#
greinke_colors = []
for row in sns.color_palette(cmap,as_cmap=False,n_colors=len(pitches)): #frequent_pitches.index)):
    greinke_colors.append([*row,1.])    
order = pd.DataFrame({"pitch": [], "colormap": [], "color": [], "marker": []})
for (pitch,mrkr,this_clr) in zip(pitches,markers,greinke_colors):   
    order = pd.concat([order,pd.DataFrame({"pitch": pitch, "color": [this_clr],"marker": mrkr, "code": pitch_type_codes[pitch]})],ignore_index=True, axis=0)

# order = make_color_order(graded=False,cmap="viridis")
order.reset_index(inplace=True)


greinke = data[data.player_name.isin(["Greinke, Zack"])].copy()

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



# greinke = data[data.player_name.isin(["Greinke, Zack"])].copy()

# # greinke["release_spin_rate"] = greinke["release_spin_rate"].fillna(0)
# greinke.loc[:,"release_speed_float"] = greinke.release_speed.astype("float64")
# greinke.loc[:,"release_spin_float"] = greinke.release_spin_rate.astype("float64")
# greinke.loc[:,"release_pos_x_float"] = greinke.release_pos_x.astype("float64")
# greinke.loc[:,"release_pos_z_float"] = greinke.release_pos_z.astype("float64")
# greinke.loc[:,"pfx_x_float"] = greinke.pfx_x.astype("float64")
# greinke.loc[:,"pfx_z_float"] = greinke.pfx_z.astype("float64")
# greinke.loc[:,"plate_x_float"] = greinke.plate_x.astype("float64")
# greinke.loc[:,"plate_z_float"] = greinke.plate_z.astype("float64")

# greinke.loc[:,"plate_x_norm"] = np.array((greinke.plate_x - (-0.71)) / (0.71 - (-0.71)),dtype="float")
# greinke.loc[:,"plate_z_norm"] = np.array((greinke.plate_z - greinke.sz_bot) / (greinke.sz_top - greinke.sz_bot), dtype="float")

#########################################
#                                       #
#         Postion prior (plate_z)       #
#                                       #
#########################################
fig = plt.figure(figsize=(1.5,4))
ax = fig.add_subplot()

# Make distribution
sns.histplot(greinke,y="plate_z_norm",bins=15,ax=ax,color='silver',stat='density',kde=True,shrink=0.9,edgecolor="w", linewidth=1.5,zorder=2)
sns.kdeplot(data=greinke, y="plate_z_norm",ax=ax,fill=False,color="dodgerblue",zorder=2)#color="darkgrey")
xpos, ypos_chest, ypos_knees = 1, 1.05, -.1 #.9875, -.02

# Add lines to label strikezone
ax.axhline(y = 1, xmin=-.01, xmax=0.85,color='firebrick',linestyle=':',clip_on=False,zorder=2,lw=2)
ax.axhline(y = 0, xmin=-.01, xmax=0.85,color='firebrick',linestyle=':',clip_on=False,zorder=2,lw=2)
# ax2.axhline(y = 0, xmin=-.01, xmax=.55,color='firebrick',linestyle=':',clip_on=False,zorder=2,lw=2)
txt1 = ax.text(xpos,ypos_chest,"Chest",color='firebrick',fontweight="bold",zorder=2)#,backgroundcolor='white')
txt2 = ax.text(xpos,ypos_knees,"Knees",color='firebrick',fontweight="bold",zorder=2)#,backgroundcolor='white')

# Make figure adjustments
ax.set_xlim(0,1.25)
ax.set_ylim(-.75,1.6)
bin_ticks=[-.5,0,0.25,0.5,0.75,1,1.5]
ax.set_yticks(ticks=bin_ticks)
ax.set_yticklabels(list(map(str, [round(val,2) for val in bin_ticks])))#['1','2','3','4','5','6','7','8'])
ax.tick_params(axis='y')#,direction='out')
ax.set_ylabel(None)
ax.xaxis.set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(os.path.join(os.getcwd(),"figures","greinke_priorposition.svg"))


#########################################
#                                       #
#         Pitch behavior priors         #
#                                       #
#########################################

# # Pitches and order
# pitches = ["Sinker","Cutter","4-Seam Fastball","Slider","Curveball","Changeup"]    
# markers = ["X","o","*","s","^","P"]#["o","o"ax_top_L.spines['bottom'].set_color(tuple(3*[0.2])),"o","o","o","o"]#,"D"]#,"v"]    
# cmap =  "colorblind"#"viridis"#
# greinke_colors = []
# for row in sns.color_palette(cmap,as_cmap=False,n_colors=len(pitches)): #frequent_pitches.index)):
#     greinke_colors.append([*row,1.])    
# order = pd.DataFrame({"pitch": [], "colormap": [], "color": [], "marker": []})
# for (pitch,mrkr,this_clr) in zip(pitches,markers,greinke_colors):   
#     order = pd.concat([order,pd.DataFrame({"pitch": pitch, "color": [this_clr],"marker": mrkr, "code": pitch_type_codes[pitch]})],ignore_index=True, axis=0)

# # order = make_color_order(graded=False,cmap="viridis")
# order.reset_index(inplace=True)

# Define figure
fig = plt.figure(figsize=(2.75,2.15))

# Add gridspec for legend
gs0 = fig.add_gridspec(nrows=1, ncols=1, left=0, right=0.8, bottom=0.925,top=1)
ax0 = fig.add_subplot(gs0[0])

# Add grid spec for data
gs1 = fig.add_gridspec(nrows=2, ncols=2, left=0.001, right=1, bottom=0.001,top=0.75,hspace=1.05,wspace=.4)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])
ax3 = fig.add_subplot(gs1[2])
ax4 = fig.add_subplot(gs1[3])

# Make custom legend
xs = np.array([0,0.25,.6,0,0.25,0.6])
ys = np.array([0.1,0.1,0.1,0.3,0.3,0.3])
for (row,xpos,ypos) in zip(order.itertuples(),xs,ys):
    ax0.plot([xpos,xpos+0.06],[ypos,ypos],color=row.color)
    ax0.text(x=xpos+0.07,y=ypos,s=row.pitch + " ({})".format(row.code),color=row.color,ha="left",va="center")
ax0.set_frame_on(False)
ax0.yaxis.set_visible(False)
ax0.xaxis.set_visible(False)

# Make barplot
plt.sca(ax1)
# sns.countplot(data=_local,x="pitch_name",hue_order=order.pitch.tolist(),palette=order.color.tolist(),**kwargs={"width":1})
countplot=sns.histplot(data=greinke[greinke.pitch_name.isin(pitches)],x="player_name",hue="pitch_name",hue_order=order.pitch.tolist(),palette=order.color.tolist(), multiple="dodge", 
              stat='density', shrink=0.8, common_norm=True)#sns.countplot(data=_local,x="player_name",hue="pitch_name",hue_order=order.pitch.tolist(),palette=order.color.tolist())

ax1.legend_.remove()
ax1.set_yticks([])
ax1.set_xticks([])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.yaxis.set_visible(False)
ax1.patch.set_alpha(0)
ax1.set_xlabel(None)
ax1.patch.set_alpha(0)
ax1.set_title("Pitch distribution",fontsize=10)
ax1.set(xlabel=None,ylabel=None)
ax1.set_ylim(0,.6)

pitch_percent = np.ceil(greinke[greinke.pitch_name.isin(pitches)].pitch_name.value_counts(normalize=True)*100)

cnt=0
# Add marker instead of label
for (patch,code,percent,clr) in zip(ax1.patches[::-1],order.code.tolist(),pitch_percent[order.pitch.tolist()], order.color.tolist()):
    # plt.scatter(x=patch._x0 + patch._width/2,y=-.05,marker=mrkr,color=clr,s=15)
    plt.text(x=patch._x0 + patch._width/2,y=-.15,s=code,ha="center",rotation=25,color=clr)#,s=15)
    plt.text(x=patch._x0 + patch._width/2,y=patch._height + 0.05,s="{}%".format(int(percent)),ha="center",rotation=25,fontsize=7,color=clr)#,s=15)
    cnt+=1


# Make velo distribution
plt.sca(ax2)
sns.kdeplot(data=greinke[greinke.pitch_name.isin(pitches)],x="release_speed_float",legend=False,hue="pitch_name",palette=order.color.tolist(),hue_order=order.pitch.tolist())
ax2.set_title("Pitch Velocity (mph)",fontsize=9)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.yaxis.set_visible(False)
ax2.patch.set_alpha(0)
ax2.set_xlim([55,105])
ax2.set(xlabel=None,ylabel=None,xticks=[60,70,80,90,100])

# Make spin distribution
plt.sca(ax3)
sns.kdeplot(data=greinke[greinke.pitch_name.isin(pitches)],x="release_spin_float",legend=False,hue="pitch_name",palette=order.color.tolist(),hue_order=order.pitch.tolist())
ax3.set_title("Spin rate (rpm)",fontsize=9)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.yaxis.set_visible(False)
ax3.patch.set_alpha(0)
ax3.set(xlabel=None,ylabel=None)

# Make z movement distribution
plt.sca(ax4)
sns.kdeplot(data=greinke[greinke.pitch_name.isin(pitches)],x="pfx_z_float",legend=False,hue="pitch_name",palette=order.color.tolist(),hue_order=order.pitch.tolist())
ax4.set_title("Vertical movement (in)",fontsize=9)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.yaxis.set_visible(False)
ax4.patch.set_alpha(0)
ax4.set(xlabel=None,ylabel=None,xticks=[-1,0,1,2])

plt.rcParams.update({'font.size': 7})

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-v2-BayesballStory.png"), dpi=300, facecolor='w', edgecolor='w', bbox_inches="tight")
fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-v2-BayesballStory.svg"))#, dpi=300, facecolor='w', edgecolor='w', bbox_inches="tight")


#########################################
#                                       #
#         Make the likelihood           #
#                                       #
#########################################

fig = plt.figure(figsize=(1.5,4))
ax = fig.add_subplot()

X = np.linspace(-.5, 1.5, 100)
_L = stats.norm.pdf(X, 0.7, 0.1)
ax.plot(_L,X,color=order[order.pitch.isin(["4-Seam Fastball"])].color.tolist()[0],zorder=3)


# Add lines to label strikezone
ax.axhline(y = 1, xmin=-.01, xmax=0.85,color='firebrick',linestyle=':',clip_on=False,zorder=2,lw=2)
ax.axhline(y = 0, xmin=-.01, xmax=0.85,color='firebrick',linestyle=':',clip_on=False,zorder=2,lw=2)
xpos, ypos_chest, ypos_knees = 1, 1.05, -.1
txt1 = ax.text(xpos,ypos_chest,"Chest",color='firebrick',fontweight="bold",zorder=2)#,backgroundcolor='white')
txt2 = ax.text(xpos,ypos_knees,"Knees",color='firebrick',fontweight="bold",zorder=2)#,backgroundcolor='white')

# Make figure adjustments
ax.set_ylim(-.75,1.6)
bin_ticks=[-.5,0,0.25,0.5,0.75,1,1.5]
ax.set_yticks(ticks=bin_ticks)
ax.set_yticklabels(list(map(str, [round(val,2) for val in bin_ticks])))#['1','2','3','4','5','6','7','8'])
ax.tick_params(axis='y')#,direction='out')
ax.set_ylabel(None)
ax.xaxis.set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(os.path.join(os.getcwd(),"figures","greinke_likelihoodposition.svg"))


#########################################
#                                       #
#         Show the full Bayes           #
#                                       #
#########################################

fig = plt.figure(figsize=(8,1.25))

# Add gridspec for legend
gs0 = fig.add_gridspec(nrows=1, ncols=1, left=0, right=0.1, bottom=0.01,top=.85)
ax0 = fig.add_subplot(gs0[0])

# Add grid spec for data
gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.2, right=.8,bottom=0.01,top=gs0.top)# bottom=0.55,top=1)#,hspace=.25)
ax1 = fig.add_subplot(gs1[0])
ax1.set_ylim(-.5,1.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlabel("Plate Position (% strike zone)",fontweight="bold")
ax1.set_ylabel("Plate Position (% strike zone)",fontweight="bold")

gs2 = fig.add_gridspec(nrows=1, ncols=1, left=gs1.left, right=gs1.right, bottom=gs1.top+0.1,top=1)#,hspace=.25)
ax2 = fig.add_subplot(gs2[0])

# Get pdf for greinke data
mean,std=stats.norm.fit(greinke.plate_z_norm)
xx = np.linspace(-.5, 1.5, 100)
greinke_prior = stats.norm.pdf(xx, mean, std)
sns.histplot(greinke,y="plate_z_norm",bins=15,ax=ax0,color='silver',stat='density',kde=False,shrink=0.9,edgecolor="w", linewidth=1.5,zorder=2)
ax0.plot(greinke_prior,xx,color="dodgerblue",zorder=3)
xpos, ypos_chest, ypos_knees = 1, 1.15, -.2 #.9875, -.02

# Add lines to label strikezone
ax0.axhline(y = 1, xmin=0, xmax=1.05,color='firebrick',linestyle=':',clip_on=False,zorder=2,lw=2)
ax0.axhline(y = 0, xmin=0, xmax=1.05,color='firebrick',linestyle=':',clip_on=False,zorder=2,lw=2)
txt1 = ax0.text(xpos,ypos_chest,"Chest",color='firebrick',fontweight="bold",zorder=2)#,backgroundcolor='white')
txt2 = ax0.text(xpos,ypos_knees,"Knees",color='firebrick',fontweight="bold",zorder=2)#,backgroundcolor='white')

# Make figure adjustments
ax0.set_xlim(0,1.25)
ax0.set_ylim(-.75,1.6)
bin_ticks=[-.5,0,0.25,0.5,0.75,1,1.5]
ax0.set_yticks(ticks=bin_ticks)
ax0.set_yticklabels(list(map(str, [round(val,2) for val in bin_ticks])))#['1','2','3','4','5','6','7','8'])
ax0.tick_params(axis='y')#,direction='out')
ax0.set_ylabel(None)
ax0.xaxis.set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)


#### AXIS 2
plt.sca(ax1)
ax1.set_xlim(0,1.2)
nn=6
gs1_1 = fig.add_gridspec(nrows=1, ncols=nn, left=gs1.left, right=gs1.right, bottom=gs1.bottom,top=gs1.top)#,hspace=.25)
ax1_1 = []
ll = []
likelihood = []
posterior = []
MAP=[]
MAPval=[]
MAPidx=[]
MAPx=[]
MAPy=[]
Lcolor="grey"
Pcolor=order[order.pitch.isin(["Curveball"])].color.tolist()[0]
MAPcolor = order[order.pitch.isin(["4-Seam Fastball"])].color.tolist()[0]

xvals=np.linspace(0,1,nn)

# Bayes
for cnt,xval in enumerate(xvals):

    # Add a subplot
    ax1_1.append(fig.add_subplot(gs1_1[cnt],facecolor=None))
    
    # Make likelihood and multiply with prior to get posterior
    _L = stats.norm.pdf(xx, xval, 0.1)
    likelihood.append(_L)
    _P = np.multiply(_L,greinke_prior)
    
    # Compute MAP and store
    ax1_1[cnt].scatter(_P.max(),xx[_P.argmax()],color=MAPcolor,s=15,zorder=5,clip_on=False)
    xlims = ax1_1[cnt].get_xlim()
    ax1_1[cnt].set_xlim(0,np.maximum(_P.max(),_L.max())+.25)
    MAPval.append(_P.max())
    MAPidx.append(xx[_P.argmax()])
    
    # Plot everything
    posterior.append(_P)
    ax1_1[cnt].plot(greinke_prior,xx,color="dodgerblue",zorder=3)
    ax1_1[cnt].plot(_L,xx,color=Lcolor,zorder=3)
    ax1_1[cnt].plot(_P,xx,color=Pcolor,zorder=4)#order[order.pitch.isin(["4-Seam Fastball"])].color.tolist()[0]
    ax1_1[cnt].axvline(x=0,color="darkgrey",linestyle="--")
    ax1_1[cnt].set_ylim(-.5,1.5)
    ax1_1[cnt].set_axis_off()

# Add label to the right
ax1_r=ax1.twinx()
ax1_r.set_ylim(-.5,1.5)
ax1_r.set_yticklabels([])
ax1_r.set_ylabel("Error Predicted",color=MAPcolor,fontweight="bold",rotation=270,x=1.05)
ax1_r.spines["right"].set_linewidth(2.0)
ax1_r.spines["right"].set_color(MAPcolor)
ax1_r.yaxis.set_label_coords(1.05,.5)
ax1_r.tick_params(axis='y', colors=MAPcolor)

# Beautify
ax0.set_title("Prior belief",fontstyle="italic",color="dodgerblue",fontsize=9,y=1.05)
colors=["dodgerblue",Lcolor,Pcolor,MAPcolor]
labels=["Prior","Likelihood","Posterior","Prediction"]
for clr,lab in zip(colors,labels):
    ax2.plot([], [], color=clr,alpha=1, label=lab)
leg = ax2.legend(loc="upper center",frameon=False,ncol=4,fontsize=9,bbox_to_anchor=(.5,2))    
ax2.set_frame_on(False)
ax2.yaxis.set_visible(False)
ax2.xaxis.set_visible(False)
for (text,clr) in zip(leg.get_texts(),colors):
    text.set_color(clr)

from matplotlib.ticker import StrMethodFormatter
ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) 
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

plt.show()
fig.savefig(os.path.join(os.getcwd(),"figures","bayes-greinke.png"), dpi=300, facecolor='w', edgecolor='w', bbox_inches="tight")
fig.savefig(os.path.join(os.getcwd(),"figures","bayes-greinke.svg"))


#########################################
#                                       #
#      Show the theoretical model       #
#                                       #
#########################################

fig = plt.figure(figsize=(4.75,2))

# Add grid spec for data
gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.01, right=.95,bottom=0.01,top=.95)# bottom=0.55,top=1)#,hspace=.25)
ax1 = fig.add_subplot(gs1[0])
ax1.set_ylim(-.75,2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlabel("Plate Position (% strike zone)",fontweight="bold",fontsize=9)
ax1.set_ylabel("Contact error (% strike zone)",fontweight="bold",fontsize=9)


# Get pdf for greinke data
mean,std=stats.norm.fit(greinke.plate_z_norm)
xx = np.linspace(-.5, 1.5, 100)
greinke_prior = stats.norm.pdf(xx, mean, std)

# Loop through positions in strike zone
xvals=np.linspace(1,0,9)
meanP=[]
allx =[]
allL =[]
for cnt,xval in enumerate(xvals):
    _L = stats.norm.pdf(xx, xval, 0.1)
    _P = np.multiply(_L,greinke_prior)
    
    _Lvals = np.random.normal(xval,0.35,100)
    _Pfit  = stats.norm.fit(_P)
    _Pvals = np.random.normal(_Pfit[0],_Pfit[1],100)
    xwithnoise = len(_Lvals)*[1-xval] + np.random.normal(.001,.01,len(_Lvals))
    ax1.scatter(xwithnoise,_Lvals,s=10,color="lightgray")
    meanP.append(_Lvals.mean())

# Fit data
X = sm.add_constant(xvals)# prepend=False)
ols = sm.OLS(meanP,X)
ols_result = ols.fit()

thisx = np.linspace(-.15,1.15, 100)
ax1.plot(thisx,len(thisx)*[0.5],color="grey",label="Mostly Likelihood")
ax1.plot(thisx,1.5 - 2*(thisx),color="black",linestyle="--",label="Mostly Prior")
ax1.plot(1-thisx,ols_result.params[0] + (ols_result.params[1]*thisx),linewidth=3, color=MAPcolor,label="Bayesian solution")

# ax1.set_yticks([])
ax1.invert_xaxis()
ax1.set_xticks([1,0.5,0])
ax1.set_yticks([])
ax1.set_xticklabels(["Chest (100 %)", "50 %", "Knees (0 %)"],fontweight="bold",fontsize=9)
ax1.get_xticklabels()[0].set_color("firebrick")
ax1.get_xticklabels()[2].set_color("firebrick")
ax1.legend(loc="upper center",frameon=False,ncol=2,fontsize=9,bbox_to_anchor=(.7,1.25))
ax1.set_xlim(1.15,-.15)

# plt.tight_layout()
fig.savefig(os.path.join(os.getcwd(),"figures","bayes_to_error.svg"), bbox_inches="tight")


#########################################
#                                       #
#     Distribution of ball contact      #
#                                       #
#########################################

fig = plt.figure(figsize=(3,1))
ax = fig.add_subplot()
# true_shift=which_data_true_shift,
              # deviation=which_data_deviation,
sns.histplot(data=data,x="vertical_contact_error",bins=10,ax=ax,color='firebrick',stat='density',kde=False,shrink=0.9,edgecolor="w", linewidth=1.5,zorder=1)
mean,std=stats.norm.fit(data.vertical_contact_error)
xx = np.linspace(-2,2, 100)
pdf = stats.norm.pdf(xx, mean, std)
# sns.kdeplot(data=data,y="vertical_contact_error",ax=ax,fill=False,color="firebrick",zorder=2)#color="darkgrey")
ax.plot(xx,pdf,color="firebrick")
ax.fill_between(xx,pdf,where=(xx<data.vertical_contact_error.min()),color="silver")
ax.yaxis.set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis='x', labelrotation=90)
ax.invert_xaxis()
ax.set_xlabel([])
# ax.set_frame_on(False)
plt.show()
fig.savefig(os.path.join(os.getcwd(),"figures","vertical_contact_error_distribution.svg"))

#########################################
#                                       #
#       Make full bayes for batter      #
#                                       #
#########################################

# Get pdf for greinke data - aka prior
mean,std=stats.norm.fit(greinke.plate_z_norm)
X= np.linspace(-.5, 1.5, 100)
greinke_prior = stats.norm.pdf(X, mean, std)

# Get likelihood
likelihood = stats.norm.pdf(X, 0.35, 0.1)

# Get posterior
posterior = np.multiply(greinke_prior,likelihood)

# Make figure
fig = plt.figure(figsize=(1.5,4))
ax = fig.add_subplot()
ax.plot(greinke_prior,X,color="dodgerblue",zorder=2,label="Prior")
# ax.fill_between(greinke_prior,X,color="dodgerblue",zorder=2,alpha=0.2)
ax.plot(likelihood,X,color=(0.5,0.5,0.5),zorder=3,label="Likelihood")
# ax.fill_between(likelihood,X,color=(0.5,0.5,0.5),zorder=3,alpha=0.2)
ax.plot(posterior,X,color=order[order.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,label="Posterior")
# ax.fill_between(posterior,X,color=order[order.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,alpha=0.2)
ax.scatter(posterior.max(),X[posterior.argmax()],s=50,color=order[order.pitch.isin(["4-Seam Fastball"])].color.tolist()[0],zorder=5,marker="*",label="Estimate")

# Make figure adjustments
ax.set_ylim(-.5,1.5)
bin_ticks=[-.5,0,0.25,0.5,0.75,1,1.5]
ax.set_yticks([])#ticks=bin_ticks)
# ax.set_yticklabels(list(map(str, [round(val,2) for val in bin_ticks])))#['1','2','3','4','5','6','7','8'])
ax.tick_params(axis='y')#,direction='out')
ax.set_ylabel(None)
ax.xaxis.set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-v2-GreinkePriorLikelihoodPosterior.svg"))

#########################################
#                                       #
#           Make P(z| pitch)            #
#                                       #
#########################################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

for pitch in pitches:
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.pitch_name.isin([pitch]),"plate_z_norm"])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std)
    ax.plot(prior,X,color=order.loc[order.pitch.isin([pitch]),"color"].tolist()[0],label="Prior")

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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-v3-PlatePositionByPitch.svg"))

#########################################
#                                       #
#             Make P(z|velo)            #
#                                       #
#########################################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

greinke.dropna(subset=['release_speed'],inplace=True)
greinke.loc[:,'release_speed_binned'] = pd.cut(greinke['release_speed'].astype("float"),bins=[greinke.release_speed.min(),80,greinke.release_speed.max()])

bin_labels = list(map(str, range(1,4))) #['1','2','3','4','5','6','7','8','9']
velo_qbinned , velo_qbins = pd.qcut(greinke['release_speed'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
greinke.loc[:,'release_speed_qbinned'] = velo_qbinned

clrs = ['black','darkgrey','grey']
# for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'release_speed_binned'].unique(),['black','grey'],['-',':'])):
for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','darkgrey'],['-',':','--'])):
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.release_speed_qbinned == bin,'plate_z_norm'])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std)
    ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} mph".format(velo_qbins[cnt],velo_qbins[cnt+1]))

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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-v3-PlatePositionByVelo.svg"))


#########################################
#                                       #
#              Make P(z|spin)           #
#                                       #
#########################################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

greinke.dropna(subset=['spin_axis'],inplace=True)


# spin_cutbins = [0,150,360]
# greinke.loc[:,'spin_axis_binned'] = pd.cut(greinke['spin_axis'].astype("float"),bins=spin_cutbins)

# clrs = ['black','darkgrey','grey']
# for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
#     # Get pdf for greinke data - aka prior
#     mean,std=stats.norm.fit(greinke.loc[greinke.spin_axis_binned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} rpm".format(spin_cutbins[cnt],spin_cutbins[cnt+1]))


bin_labels = list(map(str, range(1,4))) #['1','2','3','4','5','6','7','8','9']
spin_qbinned , spin_qbins = pd.qcut(greinke['spin_axis'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
greinke.loc[:,'spin_axis_qbinned'] = spin_qbinned

clrs = ['black','darkgrey','grey']
# for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','lightgrey'],['-','--',':'])):
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.spin_axis_qbinned == bin,'plate_z_norm'])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std)
    ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{}".format(spin_qbins[cnt],spin_qbins[cnt+1]))

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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-v3-PlatePositionBySpinAxis.svg"))



#################
#               #
#   Spin Rate   #
#               #
#################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

greinke.dropna(subset=['release_spin_rate'],inplace=True)

bin_labels = list(map(str, range(1,4))) #['1','2','3','4','5','6','7','8','9']
spinrate_qbinned , spinrate_qbins = pd.qcut(greinke['release_spin_rate'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
greinke.loc[:,'release_spin_rate_qbinned'] = spinrate_qbinned

clrs = ['black','darkgrey','grey']
# for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','lightgrey'],['-','--',':'])):
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.release_spin_rate_qbinned == bin,'plate_z_norm'])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std)
    ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} rpm".format(spinrate_qbins[cnt],spinrate_qbins[cnt+1]))

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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-v3-PlatePositionBySpinRate.svg"))


#########################
#                       #
#   Vertical Movement   #
#                       #
#########################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

greinke.dropna(subset=['pfx_z_float'],inplace=True)

# pfxz_cutbins = [greinke['pfx_z_float'].min(), 0 , greinke['pfx_z_float'].max()]
# greinke.loc[:,'pfx_z_binned'] = pd.cut(greinke['pfx_z_float'].astype("float"),bins=pfxz_cutbins)

# clrs = ['black','darkgrey']
# for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'pfx_z_binned'].unique(),['black','grey'],['-',':'])):
#     # Get pdf for greinke data - aka prior
#     mean,std=stats.norm.fit(greinke.loc[greinke.pfx_z_binned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} inches".format(pfxz_cutbins[cnt],pfxz_cutbins[cnt+1]))

bin_labels = list(map(str, range(1,4))) #['1','2','3','4','5','6','7','8','9']
pfxz_qbinned , pfxz_qbins = pd.qcut(greinke['pfx_z_float'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
greinke.loc[:,'pfx_z_qbinned'] = pfxz_qbinned

clrs = ['black','darkgrey','grey']
# for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','lightgrey'],['-','--',':'])):
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.pfx_z_qbinned == bin,'plate_z_norm'])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std)
    ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} inches".format(pfxz_qbins[cnt],pfxz_qbins[cnt+1]))

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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-v3-PlatePositionByVerticalMovement.svg"))



#########################
#                       #
#   Vertical Movement   #
#                       #
#########################

# Make figure
fig = plt.figure(figsize=(1,3.5))
ax = fig.add_subplot()

greinke.loc[:,"vertical_drop"] = greinke.loc[:,"release_pos_z_float"] - greinke.loc[:,"plate_z"]
greinke.dropna(subset=['vertical_drop'],inplace=True)

# vdrop_cutbins = [greinke['vertical_drop'].min(), 0 , greinke['pfx_z_float'].max()]
# greinke.loc[:,'pfx_z_binned'] = pd.cut(greinke['pfx_z_float'].astype("float"),bins=pfxz_cutbins)

# clrs = ['black','darkgrey']
# for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'pfx_z_binned'].unique(),['black','grey'],['-',':'])):
#     # Get pdf for greinke data - aka prior
#     mean,std=stats.norm.fit(greinke.loc[greinke.pfx_z_binned == bin,'plate_z_norm'])
#     X= np.linspace(-.5, 1.5, 100)
#     prior = stats.norm.pdf(X, mean, std)
#     ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} inches".format(pfxz_cutbins[cnt],pfxz_cutbins[cnt+1]))

bin_labels = list(map(str, range(1,5))) #['1','2','3','4','5','6','7','8','9']
vdrop_qbinned , vdrop_qbins = pd.qcut(greinke['vertical_drop'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
greinke.loc[:,'vertical_drop_qbinned'] = vdrop_qbinned

clrs = ['black','darkgrey','grey']
# for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
for cnt,(bin,clr,lnstyle) in enumerate(zip(bin_labels,['black','grey','lightgrey','darkgrey'],['-','--',':','-.'])):
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.vertical_drop_qbinned == bin,'plate_z_norm'])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std)
    ax.plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} ft".format(round(vdrop_qbins[cnt],1),round(vdrop_qbins[cnt+1],1)))

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

fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-v3-PlatePositionByVerticalDrop.svg"))

#########################################
#                                       #
#      Make scatter of spin vs velo     #
#                                       #
#########################################

fig = plt.figure(figsize=(2,1))
ax = fig.add_subplot()
ss = sns.scatterplot(data=greinke.loc[greinke.pitch_name.isin(pitches),:],y='release_speed_float', x='spin_axis',hue="pitch_name",hue_order=order.pitch.tolist(),palette=order.color.tolist(),legend=False,size=2,linewidth=.1)
ax.set_xlabel("Spin axis (deg)")
ax.set_ylabel("Release velocity (mph)")
plt.rcParams.update({'font.sans-serif':'Arial','font.size':8})
ax.set_yticks([60,70,80,90])
ax.set_ylim([55,100])
fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-v3-SpinAxisVsVelocity.svg"))
