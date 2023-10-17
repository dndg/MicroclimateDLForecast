import matplotlib.pyplot as plt
import seaborn as sns



def make_heatmap(heatmap, cmap, vmin, vmax, cbar_label, title, 
                 figsize = (10,8), aspect = 1, shrink = 0.7,
                 tick_params_labelsize = 20, ylabel_fontsize = 35, title_fontsize = 40,
                 save_figure=False, figure_filename=None):
    '''
    This method creates a heatmap using Seaborn.
    Inputs:
     - heatmap: The data to be displayed.
     - cmap: The matplotlib color map to be used.
     - vmin: The minimum value represented in the colorbar.
     - vamx: The maximum value represented in the colorbar.
     - cbar_label: The label for the colorbar.
     - title: The title of the figure.
     - figsize: The size of the figure (height, width).
     - aspect: The aspect ratio of the figure (height/width).
     - shrink:  The amount by which to shrink the colorbar.
     - tick_params_labelsize: The size of the colorbar tick labels.
     - ylabel_fontsize: The font size of the colorbar title.
     - title_fontsize: The font size of the figure title.
     - save_figure:A boolean parameter to decide whether to save the figure or not.
     - figure_filename: The filename path for the saved figure.
    '''
                     

    fig, ax = plt.subplots(1,figsize = figsize)
    ax.set_aspect(aspect)
    heat = sns.heatmap(heatmap, cmap = cmap, ax=ax, vmin = vmin, vmax = vmax,
                       cbar_kws = {'shrink':shrink, 'label': cbar_label})
    
    cbar = heat.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_params_labelsize)
    cbar.ax.set_ylabel(cbar_label,fontsize=ylabel_fontsize)
    
    ax.set_title(title, fontsize = title_fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    
    fig.tight_layout()

    if save_figure:                 
        fig.savefig(figure_filename, dpi=150)   

    return fig,ax