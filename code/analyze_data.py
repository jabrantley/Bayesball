# Define baseball radius and optimal launch angle
baseball_radius = 2.9/2
optimal_launch_angle = 25

# Dates for gathering data
start_date = '2015-04-01'
end_date   = '2021-11-01'

# Get data
if not load_data:
    # Get data using pybaseball
    data = statcast(start_dt=start_date,end_dt=end_date)
    first_total = len(data.index)
    print("Total number of pitches in raw data: ", first_total)
    
    # Sort by date and reset the index
    temp = data.sort_values(by="game_date")
    data = temp.reset_index()

    # Remove rows from exhibition and spring training
    data = data[~data['game_type'].isin(['E','S'])]

    # Dump file for faster access later
    data.to_pickle("data/statcast_data_2015-2021_raw.pkl")
    
# Read raw data
data = pd.read_pickle("data/statcast_data_2015-2021_raw.pkl")
total_pitches_raw = len(data.index)

# Add year as column
data['year']=pd.DatetimeIndex(data['game_date']).year

# Change knucklecurve to curve since its the same pitch for this analysis
data["pitch_name"].replace({"Knuckle Curve": "Curveball"},inplace=True)

# Remove rows with nan 
# data = data.dropna(subset=['release_pos_x'])
data['release_pos_x'] = data['release_pos_x'].astype(float)

# Remove rows with nan 
# data = data.dropna(subset=['release_pos_z'])
data['release_pos_z'] = data['release_pos_z'].astype(float)

data.loc[:,"pfx_z"] = data.loc[:,"pfx_z"].astype("float")
data.loc[:,"pfx_x"] = data.loc[:,"pfx_x"].astype("float")
data.dropna(subset=['release_pos_x','release_pos_z','pfx_x','pfx_z'],inplace=True)
data.loc[:,"pfx_z_inches"] = data.loc[:,"pfx_z"].astype("float")*12
data.loc[:,"pfx_z_norm"] = np.array((data.loc[:,'pfx_z'])/(data.loc[:,'sz_top'] - data.loc[:,'sz_bot']),dtype="float")

# Convert plate_x to % of strike zone
plate_x_left, plate_x_right = -0.71, 0.71
data.dropna(subset=['plate_x'],inplace=True)
data.loc[:,'plate_x_norm'] = np.array((.25-(-.25))*((data.loc[:,'plate_x']-plate_x_left)/ (plate_x_right - plate_x_left))+(-.25),dtype="float")
# data.loc[:,'plate_x_norm'] = np.array((data.loc[:,'plate_x']-plate_x_left)/ (plate_x_right - plate_x_left),dtype="float")

#########################################################################
#                                                                       #
#                      Calculate and bin plate_z                        #
#                                                                       #
#########################################################################
# Clean up plate z - make vals < 0 --> nan and remove
data.loc[:,'plate_z'] = data.loc[:,'plate_z'].mask(data.loc[:,'plate_z']<0)
data.dropna(subset=['plate_z'],inplace=True)

# Convert plate_z crossing to % of strikezone
data.loc[:,'plate_z_norm'] = np.array((data.loc[:,'plate_z']-data.loc[:,'sz_bot']) / (data.loc[:,'sz_top'] - data.loc[:,'sz_bot']),dtype="float")

# Bin plate z percent data - clean up first to remove outliers
data = data[(data.loc[:,"plate_z_norm"] >= np.percentile(data.loc[:,"plate_z_norm"],0.1)) & (data.loc[:,"plate_z_norm"] <= np.percentile(data.loc[:,"plate_z_norm"],99.9))]
data.dropna(subset=['plate_z_norm'],inplace=True)

# Define bin labels and bin by # of percetiles
bin_labels = list(map(str, range(1,10))) #['1','2','3','4','5','6','7','8','9']
plate_z_norm_qbinned , plate_z_norm_qbins = pd.qcut(data['plate_z_norm'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
data['plate_z_norm_qbinned'] = plate_z_norm_qbinned

# Compute mean of bin from bin edges
plate_z_norm_qbins_mean = []
for idx in range(len(plate_z_norm_qbins)-1):
    plate_z_norm_qbins_mean.append(round(np.mean( [ plate_z_norm_qbins[idx] , plate_z_norm_qbins[idx+1] ] ),3))
    
plate_z_norm_qbinned_percentile      = []
plate_z_norm_qbinned_percentile_mean = []
for val in data['plate_z_norm_qbinned']:
    plate_z_norm_qbinned_percentile_mean.append(plate_z_norm_qbins_mean[int(val)-1])
    plate_z_norm_qbinned_percentile.append(plate_z_norm_qbins[int(val)-1])

data.loc[:,'plate_z_norm_qbinned_percentile']      = np.array(plate_z_norm_qbinned_percentile,dtype="float")
data.loc[:,'plate_z_norm_qbinned_percentile_mean'] = np.array(plate_z_norm_qbinned_percentile_mean,dtype="float")

#########################################################################
#                                                                       #
#                      Calculate contact error                          #
#                                                                       #
#########################################################################

data['launch_angle'] = data['launch_angle'].astype(float)
data.dropna(subset=["launch_angle"],inplace=True)

# Calculate contact error
data.loc[:,'vertical_contact_error'] = np.array(-baseball_radius*(np.sin (-optimal_launch_angle* np.pi / 180) - np.sin(-data.loc[:,'launch_angle']* np.pi / 180)),dtype="float")
data.dropna(subset=["vertical_contact_error"],inplace=True)

# Convert contact error to % of strikezone
data.loc[:,'vertical_contact_error_norm'] = np.array(data.loc[:,'vertical_contact_error']  / (12*(data.loc[:,'sz_top'] - data.loc[:,'sz_bot'])),dtype="float") #  (baseball_radius*2)
data.dropna(subset=["vertical_contact_error_norm"],inplace=True)



###########################################
#                                         #
#      Which data to use for analysis     #
#                                         #
###########################################
which_data_true_shift = "plate_z_norm"
exec("%s = %s" % ("which_bins",which_data_true_shift + "_qbins") )
exec("%s = %s" % ("which_bins_mean",which_data_true_shift + "_qbins_mean") )
which_data_deviation ="vertical_contact_error_norm"


#########################################################################
#                                                                       #
#                          Process Pitching                             #
#                                                                       #
#########################################################################

# Create dictionary for matching pitch names with codes

pitch_type_codes = {"Sinker"         : "SI",
                    "Changeup"       : "CH",
                    "Slider"         : "SL",
                    "4-Seam Fastball": "FF",
                    "Knuckle Curve"  : "KC",
                    "Curveball"      : "CU",
                    "2-Seam Fastball": "FT",
                    "Cutter"         : "FC",
                    "Split-Finger"   : "FS",
                    "Splitter"       : "FS",
                    "Pitch Out"      : "PO",
                    "Eephus"         : "EP",
                    "Forkball"       : "FO",
                    "Knuckleball"    : "KN",
                    "Fastball"       : "FA",
                    "Screwball"      : "SC",
                    "nan"            : "UN",
                    " "              : " "
                   }

# Get all pitches
all_pitches  = data.pitch_name.value_counts()
frequent_pitches = all_pitches[all_pitches > 10000]
which_pitches = frequent_pitches.index
    
# Pre-allocate variables    
const   = []
means   = []
se_mean = []
se_lo   = []
se_hi   = []
err_at_middle = []
pitch_codes = []

# Run regression 
for pitch in which_pitches:
    # Get pitch and associated data
    temp = data[data["pitch_name"].isin([pitch])]
    ols_result = run_regression(temp[which_data_true_shift + "_qbinned_percentile"],temp[which_data_deviation])

    
    # Append results to lists
    pitch_codes.append( list(pitch_type_codes.values())[list(pitch_type_codes.keys()).index(pitch)] )    
    const.append(ols_result.params[0])
    means.append(ols_result.params[1])
    se_mean.append([ols_result.bse[0],ols_result.bse[1]])
    se_lo.append(ols_result.params[1] - ols_result.bse[0])
    se_hi.append(ols_result.params[1] + ols_result.bse[1])
    err_at_middle.append(ols_result.params[0] + ols_result.params[1] * (0))
    
    
# Compute regression using statmodels.OLS for ALL PITCHES in which_pitches
temp = data[data.pitch_name.isin(which_pitches)]
xx_all = np.array(temp[which_data_true_shift + "_qbinned_percentile_mean"])#_qbinned_percent_mean
yy_all = np.array(temp[which_data_deviation])
ols_result_all = run_regression(xx_all,yy_all)
prstd_ols_all, iv_l_ols_all, iv_u_ols_all = wls_prediction_std(ols_result_all) # for getting confidence intervals

# X_all = sm.add_constant(xx_all)#, prepend=False)
# ols_all = sm.OLS(yy_all,X_all)
# ols_result_all = ols_all.fit()

# Append results of all pitches to list
pitch_codes.append("All")    
const.append(ols_result_all.params[0])
means.append(ols_result_all.params[1])
se_mean.append([ols_result_all.bse[0],ols_result_all.bse[1]])
se_lo.append(ols_result_all.params[1] - ols_result_all.bse[0])
se_hi.append(ols_result_all.params[1] + ols_result_all.bse[1])
err_at_middle.append(ols_result_all.params[0] + ols_result_all.params[1] * (0.5))

# Add to dataframe for easy manipulation and plotting
pitch_list = which_pitches.values.tolist()
pitch_list.append('All')
results_by_pitch = pd.DataFrame({'pitch': pitch_list,
                                 'pitch_code': pitch_codes,
                                 'const': const,
                                 'means': means,
                                 'absmeans': [-1*ii for ii in means],
                                 'se_mean': se_mean,
                                 'se_mean_lo': se_lo,
                                 'se_mean_hi': se_hi,
                                 'error_at_mid': err_at_middle})
