########################################################################
#                                                                      #
#        MAKE FIGURE - variance of each pitch type.                    #
#                                                                      #
########################################################################
# For each pitch type per pitcher estimate the variance in vertical axis. 
# Consider how slope/bias relates to width of distribution, i.e., uncertainty
min_pitches = 200

# Create arrays to store data for adding to dataframe
num_pitches   = []
pitcher_name  = []
pitcher_num   = []
pitch_name    = []
pitch_code    = []
year          = []
ols_const     = []
ols_slope     = []
ols_se_mean   = []
ols_se_lo     = []
ols_se_hi     = []
err_at_middle = []
err_at_bottom = []
err_at_top    = []
pfx_z_mean    = []
pfx_z_med     = []
pfx_z_std     = []
error_mean    = []
error_med     = []
error_std     = []
error_max     = []
pitch_mean    = []
pitch_med     = []
pitch_std     = []

# Group data by pitcher - each year they pitched will be mixed up

group_by_pitcher = data.groupby('pitcher')
print("Performing regression for each pitch and for each individual pitcher ... ")
# for each of the pitchers
for this_pitcher in tqdm(group_by_pitcher.pitcher.unique().index.tolist()):
        # Group by the pitcher
        group_by_pitch = group_by_pitcher.get_group(this_pitcher).groupby('pitch_name')

        # for each pitch type
        for this_pitch in group_by_pitch.pitch_name.unique().index.tolist():
            local_data = group_by_pitch.get_group(this_pitch)
            num_pitches.append(len(local_data))
            if len(local_data) > min_pitches:
                # Store pitcher info
                pitcher_num.append(this_pitcher)
                pitcher_name.append(local_data.player_name.unique()[0])
                pitch_name.append(this_pitch)
                pitch_code.append(local_data.pitch_type.unique()[0])
                # year.append(this_year)
                pfx_z_mean.append(local_data.pfx_z_norm.mean())
                pfx_z_med.append(local_data.pfx_z_norm.median())
                pfx_z_std.append(local_data.pfx_z_norm.std())

                # Fit data
                xx = np.array(local_data[which_data_true_shift])#_qbinned_percent_mean
                yy = np.array(local_data[which_data_deviation])
                ols_result = run_regression(xx,yy)
                
                # Append fit data to lists
                ols_const.append(ols_result.params[0])
                ols_slope.append(ols_result.params[1])
                ols_se_mean.append([ols_result.bse[0], ols_result.bse[1]])
                ols_se_lo.append(ols_result.params[1] - ols_result.bse[0])
                ols_se_hi.append(ols_result.params[1] + ols_result.bse[1])
                err_at_middle.append(ols_result.params[0] + ols_result.params[1] * (0.5))
                err_at_bottom.append(ols_result.params[0] + ols_result.params[1] * (0.0))
                err_at_top.append(ols_result.params[0] + ols_result.params[1] * (1))
                
                # Get stats for pitch location and spread
                pitch_mean.append(local_data[which_data_true_shift].mean())
                pitch_med.append(local_data[which_data_true_shift].median())
                pitch_std.append(local_data[which_data_true_shift].std())
                
                # Get stats for error
                error_mean.append(local_data[which_data_deviation].mean())
                error_med.append(local_data[which_data_deviation].median())
                error_std.append(local_data[which_data_deviation].std())
                error_max.append(local_data[which_data_deviation].max())
                
data_by_pitcher = pd.DataFrame({"pitcher": pitcher_num,
                                "player_name": pitcher_name,
                                "pitch_name": pitch_name,
                                "pitch_type": pitch_code,
                                #"year": year,
                                "pfx_z_mean": np.array(pfx_z_mean),
                                "pfx_z_med": np.array(pfx_z_med),
                                "pfx_z_std": np.array(pfx_z_std),
                                "ols_const": np.array(ols_const),
                                "ols_slope": np.array(ols_slope),
                                "ols_slope_abs": -np.array(ols_slope),
                                "ols_se_mean": ols_se_mean,
                                "ols_se_lo": np.array(ols_se_lo),
                                "ols_se_hi": np.array(ols_se_hi),
                                "err_at_middle": np.array(err_at_middle),
                                "err_at_bottom": np.array(err_at_bottom),
                                "err_at_top": np.array(err_at_top),
                                "pitch_mean": np.array(pitch_mean),
                                "pitch_med": np.array(pitch_med),
                                "pitch_std": np.array(pitch_std),
                                "error_mean": np.array(error_mean),
                                "error_med": np.array(error_med),
                                "error_std": np.array(error_std),
                                "error_max": np.array(error_max)})

data_by_pitcher_trim = data_by_pitcher[data_by_pitcher.pitch_name.isin(which_pitches)].copy()