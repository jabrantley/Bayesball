########################################################################
#                                                                      #
#                  MAKE FIGURE SHOWING RELEASE/ETC                     #
#                                                                      #
########################################################################

# This is the original Figure 1, which has since been removed and replaced. 
# Keeping it here for reference and to show how its built almost entirely in python

def get_color_list(cmap="viridis"):
    these_colors = []
    for row in sns.color_palette(cmap,as_cmap=False,n_colors=len(which_pitches)): #frequent_pitches.index)):
        these_colors.append([*row,1.])
    return these_colors
    
def make_cmap(clr="Reds"):
    return cm.get_cmap(plt.get_cmap(clr))(np.linspace(0.0, 1.0, 100))[np.newaxis, :, :3]
  
def make_color_order(graded=True,cmap="viridis"):
    # Graded means a color map is applied to for each color so that KDE plots have a gradient. Otherwise it is a fixed color
    markers = ["X","^","*","s","o","P","D","v"]
    order = pd.DataFrame({"pitch": [], "colormap": [], "color": [], "marker": []})
    if graded:
        cmaps = ["Greys", "Blues", "Reds", "Greens", "Purples", "Oranges", "RdPu"]
        for cnt,(pitch, clr, mrkr) in enumerate(zip(which_pitches,cmaps,markers[:len(which_pitches)])):   
            local_cmap = make_cmap(clr=clr)
            order = pd.concat([order,pd.DataFrame({"pitch": pitch, "colormap": clr, "color": [local_cmap[0][50]],"marker": mrkr})],ignore_index=True, axis=0)
    else:
        cmaps = len(markers)*[cmap]
        for cnt,(pitch, clr, mrkr,this_clr) in enumerate(zip(which_pitches,cmaps,markers[:len(which_pitches)],get_color_list(cmap=cmap))):   
            order = pd.concat([order,pd.DataFrame({"pitch": pitch, "colormap": clr, "color": [this_clr],"marker": mrkr})],ignore_index=True, axis=0)

    return order

light_gray = tuple(3*[225/255])
dark_grey = tuple(3*[0.3])

# Define pitch colors, markers, etc
pitches = ["Cutter","4-Seam Fastball","Slider", "Changeup","Sinker","Curveball",]    
markers = ["X","o","*","s","^","P"]#["o","o"ax_top_L.spines['bottom'].set_color(tuple(3*[0.2])),"o","o","o","o"]#,"D"]#,"v"]    
colors  = sns.color_palette("viridis",as_cmap=False,n_colors=len(pitches))#[(32/255,242/255,182/255), (31/255,123/255,254/255), (249/255,15/255,18/255), (254/255,126/255,0/255), (93/255,247/255,2/255), (185/255,1/255,7/255)]#
order = pd.DataFrame({"pitch": [], "colormap": [], "color": [], "marker": []})
for (pitch, clr, mrkr,this_clr) in zip(pitches,colors,markers,colors):   
    order = pd.concat([order,pd.DataFrame({"pitch": pitch, "colormap": [clr], "color": [this_clr],"marker": mrkr})],ignore_index=True, axis=0)

# order = make_color_order(graded=False,cmap="viridis")
order.reset_index(inplace=True)

# Which pitchers to show                               
pitchers = ["Scherzer, Max", "Morton, Charlie", "Verlander, Justin"]

###########################################
#                                         #
#   Define figure layout using gridspec   #
#                                         #
###########################################
fig = plt.figure(constrained_layout=False)
fig.set_size_inches(7.5,8)

# Top panel
gs_top = fig.add_gridspec(nrows=1, ncols=2, left=0.1, right=0.75,bottom=0.9,top=.99)#,width_ratios=[1,1,1])
ax_top_L = fig.add_subplot(gs_top[0,0],facecolor="None",)
ax_top_C = fig.add_subplot(gs_top[0,1],facecolor="None")
# ax_top_R = fig.add_subplot(gs_top[0,2],facecolor="None")

# Middle panel
img_ratio = 1440/684
xx,yy = 0.01,0.95
height = (yy-xx)*img_ratio
gs_mid = fig.add_gridspec(nrows=1, ncols=1, left=0.1, right=0.99,bottom=0.455,top=.825)
ax_mid = fig.add_subplot(gs_mid[0,0],facecolor="None",visible=True)#False)

# Bottom panel
nrows, ncols = len(pitchers), 4
gs_bot = fig.add_gridspec(nrows=nrows, ncols=ncols, left=0.15, right=0.99,top=.4,bottom=0.01,wspace=0.1,hspace=0.25,width_ratios=[0.5,1,1.2,1.75])
# ax_bot=[]
ax=[]
for col in range(ncols):
    for row in range(nrows):
        # ax_bot.append(fig.add_subplot(gs_bot[col,row]))
        ax.append(fig.add_subplot(gs_bot[row,col],facecolor="None"))
        
#############################
#                           #            
#    Make top left panel    #
#                           #
#############################
plt.sca(ax_top_L)
data.loc[:,"release_speed"] = data.loc[:,"release_speed"].astype("float")
sns.kdeplot(ax=ax_top_L,data=data[data.pitch_name.isin(pitches)],x="release_speed",hue="pitch_name",fill=False,hue_order=order.pitch.tolist(),legend=False,palette=order.color.tolist())#,kde=True)
ax_top_L.set_xlim([65,105])
ax_top_L.yaxis.set_visible(False)
ax_top_L.spines['right'].set_color('none')
ax_top_L.spines['left'].set_color('none')
ax_top_L.spines['top'].set_color('none')
ax_top_L.set_title("Velocity (mph)",y=1.05,color=dark_grey,fontweight="bold")
ax_top_L.xaxis.label.set_visible(False)
ax_top_L.spines['bottom'].set_color(dark_grey)
ax_top_L.xaxis.label.set_color(dark_grey)
ax_top_L.tick_params(axis='x', colors=dark_grey)

for (line,clr,mrkr) in zip(ax_top_L.lines,colors[::-1],markers[::-1]):
    plt.scatter(x=line._x[line._y.argmax()],y=line._y[line._y.argmax()]+0.05+0.0025*np.random.randn(1),marker=mrkr,color=clr,s=15)#+0.01*np.random.randn(1)

#############################
#                           #            
#   Make top center panel   #
#                           #
#############################    
plt.sca(ax_top_C)
data.loc[:,"release_spin_rate"] = data.loc[:,"release_spin_rate"].astype("float")
ax_top_C = sns.kdeplot(ax=ax_top_C,data=data[data.pitch_name.isin(pitches)],x="release_spin_rate",hue="pitch_name",fill=False,hue_order=order.pitch.tolist(),palette=order.color.tolist())#,kde=True)
ax_top_C.set_xlim([500,3500])
ax_top_C.set_xticks([1000,2000,3000])
ax_top_C.yaxis.set_visible(False)
ax_top_C.spines['right'].set_color('none')
ax_top_C.spines['left'].set_color('none')
ax_top_C.spines['top'].set_color('none')
ax_top_C.set_title("Spin Rate (rpm)",y=1.05,color=dark_grey,fontweight="bold")
ax_top_C.xaxis.label.set_visible(False)
ax_top_C.spines['bottom'].set_color(dark_grey)
ax_top_C.xaxis.label.set_color(dark_grey)
ax_top_C.tick_params(axis='x', colors=dark_grey)

for (line,clr,mrkr) in zip(ax_top_C.lines,colors[::-1],markers[::-1]):
    plt.scatter(x=line._x[line._y.argmax()],y=line._y[line._y.argmax()]+0.0002+0.0001*np.random.randn(1),marker=mrkr,color=clr,s=15)#+0.01*np.random.randn(1)
    
for row in order.itertuples():
    ax_top_C.scatter([], [], color=row.color, marker=row.marker, alpha=1, label=row.pitch)
ax_top_C.legend(loc="center left",bbox_to_anchor=(1.05, 0.50),frameon=False)    
ax_in = []


########################################################
#                                                      #            
#   Get data for each pitcher, plot in bottom panels   #
#                                                      #
########################################################
for count, pitcher in enumerate(pitchers):
    
    # Define data for first column
    _local = data[data.player_name.isin([pitcher])].copy()
    
    
    for cnt, val in enumerate(["release_pos_x","release_pos_z","pfx_x","pfx_z"]): 
        _local.loc[:,val + "_zscore"] =  np.array((_local.loc[:,val] - _local.loc[:,val].mean()) / _local.loc[:,val].std(),dtype="float")
        # _local.loc[:,val + "_zscore"].astype("float")
        
    # _local[["release_pos_x","release_pos_z","pfx_z"]] = _local[["release_pos_x","release_pos_z"]].apply(zscore)
    _grouped = _local.groupby("pitch_name")
    
    plt.sca(ax[count])
    
    # sns.countplot(data=_local,x="pitch_name",hue_order=order.pitch.tolist(),palette=order.color.tolist(),**kwargs={"width":1})
    countplot=sns.histplot(data=_local,x="player_name",hue="pitch_name",hue_order=order.pitch.tolist(),palette=order.color.tolist(), multiple="dodge", 
                  stat = 'density', shrink = 0.8, common_norm=True)#sns.countplot(data=_local,x="player_name",hue="pitch_name",hue_order=order.pitch.tolist(),palette=order.color.tolist())
    # ax[count].axis("off")
    ax[count].legend_.remove()
    ax[count].set_yticks([])
    ax[count].set_xticks([])
    ax[count].set_frame_on(False)
    ax[count].set_xlabel(None)
    splitname = pitcher.split(",")
    ax[count].set_ylabel(splitname[1] + "\n" + splitname[0],rotation=0,ha="right",x=-.075,y=0.25,fontweight="bold",color=dark_grey)

    # Add marker instead of label
    for (patch,clr,mrkr) in zip(ax[count].patches,colors[::-1],markers[::-1]):
        plt.scatter(x=patch._x0 + patch._width/2,y=-.05,marker=mrkr,color=clr,s=15)
    
    # Plot KDE and scatter on top
    plt.sca(ax[count+nrows])
    if (_local.p_throws.unique()[0] == "L"):
        _local.loc[:,"release_pos_x_zscore"] = -1*_local.loc[:,"release_pos_x_zscore"]
    sns.kdeplot(data=_local, x="release_pos_x_zscore",y="release_pos_z_zscore",levels=5,legend=False,fill=False,color=(0.9, 0.9, 0.9))
    for (name, group), clr, mrkr in zip(_grouped,order.color.tolist(),order.marker.tolist()):
        _meaned = group[["release_pos_x_zscore","release_pos_z_zscore"]].mean()
        plt.scatter(x=_meaned.release_pos_x_zscore,y=_meaned.release_pos_z_zscore,color=clr,marker=mrkr,zorder=10,s=15)
    ax[count+nrows].axis("off")
 
    # Get position 
    ax_pos = ax[count+nrows].get_position()
    ax[count+nrows].set_position([ax_pos.x0+0.0275,ax_pos.y0, ax_pos.width, ax_pos.height])

    # Get position for image
    width, height = 0.55, 0.55
    xlim, ylim = ax[count+nrows].get_xlim(), ax[count+nrows].get_ylim()
    x_start, y_start = (_local["release_pos_x_zscore"].median()-xlim[0])/(xlim[1]-xlim[0]) - .1 , (_local["release_pos_z_zscore"].median()-ylim[0])/(ylim[1]-ylim[0])-height
    
    # Add image of pitcher's arm
    axins = ax[count+nrows].inset_axes((x_start, y_start, width, height))
    img = mpimg.imread('images/pitcher_arm.png')
    axins.imshow(img)
    axins.axis("off")
    ax_in.append(axins)

    # Get movement data
    plt.sca(ax[count+2*nrows])
    if (_local.p_throws.unique()[0] == "L"):
        _local.loc[:,"pfx_x"] = -1*_local.loc[:,"pfx_x"]
    
    sns.kdeplot(data=_local[_local.pitch_name.isin(pitches)], x="pfx_x",y="pfx_z",hue="pitch_name",levels=7,fill=False,hue_order=order.pitch.tolist(),palette=order.color.tolist(),legend=False,zorder=3,alpha=0.7)
    
    # Clean up axis
    ax[count+2*nrows].spines['left'].set_position('center')
    ax[count+2*nrows].spines['bottom'].set_position('center')
    ax[count+2*nrows].spines['right'].set_color('none')
    ax[count+2*nrows].spines['top'].set_color('none')
    ax[count+2*nrows].yaxis.set_visible(False)
    ax[count+2*nrows].xaxis.set_visible(False)
    ax[count+2*nrows].spines['left'].set_color(light_gray)
    ax[count+2*nrows].spines['bottom'].set_color(light_gray)
    ax[count+2*nrows].grid(b=True, which='major', color=light_gray, linestyle='-')
    ax_pos = ax[count+2*nrows].get_position()
    ax[count+2*nrows].set_position([ax_pos.x0+0.04,ax_pos.y0, ax_pos.width, ax_pos.height])

    # Add final location in x-zone
    plt.sca(ax[count+3*nrows])
    sns.kdeplot(data=_local, x="plate_x_norm",y="plate_z_norm",level=5,legend=False,fill=False,color=(0.9, 0.9, 0.9))
    for (name, group), clr, mrkr in zip(_grouped,order.color.tolist(),order.marker.tolist()):
        n_samples = 8
        if len(group) < n_samples:
            _meaned = group[["plate_x_norm","plate_z_norm"]]
        else:
            _meaned = group[["plate_x_norm","plate_z_norm"]].sample(n_samples)
        
        plt.scatter(x=_meaned.plate_x_norm,y=_meaned.plate_z_norm,color=clr,marker=mrkr,zorder=10,s=10)
        
    # Draw srikezone
    ax[count+3*nrows].plot((-.25,.25),(-.5,-.5),'k-',linewidth=1.5,clip_on=False,color=dark_grey)
    ax[count+3*nrows].add_patch(Rectangle((-.25,0),width=.5,height=1, zorder=3,fill=False,facecolor=None,edgecolor=dark_grey,clip_on=False))    
    ax[count+3*nrows].axis("off")
    ax[count+3*nrows].set_xlim([-1,1])
    ax[count+3*nrows].set_ylim([-.35,1.5])
        
# Add batter to bottom right panel        
ax_batter = ax[count+3*nrows].inset_axes((.55,-.3,.6,1.45))
img = mpimg.imread('images/batter.png')   
ax_batter.imshow(img,zorder=1)
ax_batter.axis('off')
ax_batter.set_xlim(1100,120)
ax_batter.set_ylim(2500,90)

# Add home plate
ax[count+3*nrows].text(0,-.75,"Home plate",fontsize=9,ha="center",zorder=3,color=dark_grey)

# Put modified LMU Image in middle section
img = mpimg.imread('images/LMU-pitchmovement-text-removed-newcolors.png')   
ax_mid.imshow(img)
ax_mid.axis('off')
ax_xlims, ax_ylims = ax_mid.get_xlim(), ax_mid.get_ylim()

# Add text for each pitch type
pos_arr = [600,1600,2625,4350,5355,6260]
for (pitch, clr, mrkr, xpos) in zip(pitches,colors,markers,pos_arr):
    ax_mid.text(xpos,-75,pitch,fontweight="bold",fontsize=9,color=clr,ha="center",va="center")

# Add label for each panel. All positions are w.r.t ax_mid    
ax_mid.text(-375,-1500,"A.",fontweight="bold",fontsize=16)
ax_mid.text(2500,-1500,"B.",fontweight="bold",fontsize=16)
ax_mid.text(-375,0,"C.",fontweight="bold",fontsize=16)
ax_mid.text(-375,3750,"D.",fontweight="bold",fontsize=16)
ax_mid.text(1250,3750,"E.",fontweight="bold",fontsize=16)
ax_mid.text(3000,3750,"F.",fontweight="bold",fontsize=16)
ax_mid.text(5000,3750,"G.",fontweight="bold",fontsize=16)
ax_mid.plot((-50,7000),(4850,4850),c=light_gray,linewidth=.5,clip_on=False)
ax_mid.plot((-50,5500),(6150,6150),c=light_gray,linewidth=.5,clip_on=False)
ax_mid.set_xlim(ax_xlims)
ax_mid.set_ylim(ax_ylims)

# Set axis titles
ax[0].set_title("Pitch Usage",color=dark_grey,fontweight="bold",fontsize=9,x=0.35)
ax[3].set_title("Release Point",color=dark_grey,fontweight="bold",fontsize=9)
ax[6].set_title("Pitch Movement",color=dark_grey,fontweight="bold",fontsize=9)
ax[9].set_title("Final Position",color=dark_grey,fontweight="bold",fontsize=9)

plt.rcParams.update({'font.sans-serif':'Arial'})

# Save figure
fig.savefig(os.path.join(os.getcwd(),"figures","Figure1-V1","Figure1-pitchbehavior-viridis.png"), dpi=300, facecolor='w', edgecolor='w', bbox_inches="tight")

