########################################################################
#                                                                      #
#                            FIGURE FUNCTIONS                          #
#                                                                      #
########################################################################

# Helper functions from tutorial on customizing violin plots. 
# https://matplotlib.org/stable/gallery/statistics/customized_violin.html

def get_color_list(cmap="plasma"):
    these_colors = []
    for row in sns.color_palette(cmap,as_cmap=False,n_colors=len(which_pitches)): #frequent_pitches.index)):
        these_colors.append([*row,1.])
    return these_colors

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    
    
def my_violinplot(local_data=[],true_shift=[],deviation=[],ax=[],which_pitches=[],bin_locs=[],widths=.1):
    
    # Create new axis if none is provided
    if ax is None:
        ax = plt.gca()
        
    # Convert data into numpy array with different column sizes (http://www.asifr.com/transform-grouped-dataframe-to-numpy.html)
    xt  = local_data[local_data.pitch_name.isin(which_pitches)].loc[:,[deviation]].values
    g   = local_data[local_data.pitch_name.isin(which_pitches)].reset_index(drop=True).groupby(true_shift + "_qbinned")
    xtg = [xt[i.values,:] for k,i in g.groups.items()]
    xout = np.array(xtg,dtype=object)

    # Make violin plot
    parts = plt.violinplot(xout,positions=bin_locs, showmeans=False, showmedians=False,
            showextrema=False,widths=widths)

    # Customize - mostly copied from https://matplotlib.org/stable/gallery/statistics/customized_violin.html
    for pc in parts['bodies']:
        pc.set_facecolor('lightgrey')
        pc.set_edgecolor('darkgrey')
        pc.set_alpha(1)
        pc.set_zorder(2)

        # pc.set_facecolor((0.7,0.7,0.7))
        # pc.set_edgecolor('black')
        # pc.set_alpha(1)
    
    
    # Customize violins
    quartile1, medians, quartile3 = [], [], []
    for arr in xout:
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        quartile1.append(q1)
        medians.append(np.mean(arr))
        quartile3.append(q3)
    whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(xout, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    iqr_color=(0.5,0.5,0.5)#"grey"
    s1   = ax.scatter(bin_locs, medians, marker='o', color='white', s=15, zorder=3)
    ln1  = ax.vlines(bin_locs, quartile1, quartile3, color=iqr_color, linestyle='-', lw=5)
    s2   = ax.scatter(bin_locs, quartile3, marker='o', color=iqr_color, s=12, zorder=3)
    s3   = ax.scatter(bin_locs, quartile1, marker='o', color=iqr_color, s=12, zorder=3)
    ln2  = ax.vlines(bin_locs, whiskers_min, whiskers_max, color=iqr_color, linestyle='-', lw=1)
        
    return ax, parts

def run_regression(xx,yy):
    X = sm.add_constant(xx)#, prepend=False)
    ols = sm.OLS(yy,X)
    ols_result = ols.fit()
    return ols_result