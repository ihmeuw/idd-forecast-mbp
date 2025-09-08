__all__ = [
    "create_outcome_colormap",
    "create_diverging_colors",
    "create_change_colormap",
    "get_colors"
]

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, LogLocator
import copy
import os
import xarray as xr
import numpy as np
import geopandas as gpd
import rasterio as rio
from datetime import datetime
import matplotlib.colors as colors
from pathlib import Path
from matplotlib.patches import Rectangle
from tqdm.notebook import tqdm
from idd_forecast_mbp import constants as rfc
import matplotlib.gridspec as gridspec
from idd_forecast_mbp.helper_functions import read_income_paths
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids
from idd_forecast_mbp.xarray_functions import read_netcdf_with_integer_ids, convert_to_xarray, write_netcdf

def get_colors(n_bins, cmap_name='Reds'):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / (n_bins - 1)) for i in range(n_bins)]

def create_outcome_colormap(plot_dict):
    """Create colormap and bins for suitability values (0-365 days)."""
    bins = plot_dict['bin_dict']['bins']
    colors_dict = plot_dict['colors_dict']
    cmap_name = colors_dict['base_cmap']
    drop_num=colors_dict['drop_num']
    force_white=colors_dict['force_white']
    if cmap_name is None:
        cmap_name = 'Reds'
    n_bins = len(bins) - 1
    bin_colors = get_colors(n_bins + drop_num, cmap_name=cmap_name)
    bin_colors = bin_colors[drop_num:]
    if force_white:
        bin_colors[0] = '#ffffff'
   
    plot_dict['bin_dict']['n_bins'] = len(bins) - 1
    plot_dict['bin_dict']['bin_colors'] = bin_colors
    cmap = ListedColormap(bin_colors)
    plot_dict['bin_dict']['cmap'] = cmap
    plot_dict['bin_dict']['norm'] = BoundaryNorm(bins, cmap.N, clip = True)

def create_diverging_colors(n_bins, cmap_name='RdBu_r'):
    """
    Create a diverging color palette by generating n_bins colors
    
    Parameters:
    - n_bins: int, desired number of final colors
    - cmap_name: str, name of the matplotlib colormap to use
    
    Returns:
    - list of color tuples for the final colormap
    """
    if cmap_name is None:
        cmap_name = 'RdBu_r'
    # Get the full colormap
    cmap_full = plt.get_cmap(cmap_name)
    
    # Create n_bins colors
    bin_colors = [cmap_full(i / (n_bins - 1)) for i in range(n_bins)]
        
    return bin_colors

def create_change_colormap(plot_dict, clip_neg = False, clip_pos = False):
    """Create colormap and bins for change/difference values."""
    bins = plot_dict['bin_dict']['bins']
    colors_dict = plot_dict['colors_dict']
    cmap_name = colors_dict['base_cmap']
    remove_middle = colors_dict['remove_middle']
    force_white = colors_dict['force_white']
    if cmap_name is None:
        cmap_name = 'RdBu_r'
    n_bins = len(bins) - 1
    if remove_middle:
        bin_colors = create_diverging_colors(n_bins + 2, cmap_name=cmap_name)
        mid_index = (n_bins + 2) // 2
        if force_white:
            # there are an odd number of colors. We want to set the middle one to white and delete the colors above and below it
            bin_colors = bin_colors[:(mid_index - 1)] + ["#ffffff"] + bin_colors[(mid_index+1):]
            # print("Colors after edit:", bin_colors)
        else:
            bin_colors = bin_colors[:(mid_index - 1)] + bin_colors[mid_index+1:]
    else:
        bin_colors = create_diverging_colors(n_bins, cmap_name=cmap_name)
        if force_white:
            # there are an odd number of colors. We want to set the middle one to white and delete the colors above and below it
            mid_index = n_bins // 2
            bin_colors[mid_index] = '#ffffff'
    
    if clip_neg:
        # find all bins that are negative
        neg_bins = [i for i, b in enumerate(bins) if b < 0]
        # Remove all negative bins except for the first one
        # Track which ones are removed so we can remove corresponding colors
        if len(neg_bins) > 1:
            bins = np.delete(bins, neg_bins[1:])
            bin_colors = [color for i, color in enumerate(bin_colors) if i not in neg_bins[1:]]
        plot_dict['bin_dict']['bins'] = bins
    if clip_pos:
        # find all bins that are negative
        pos_bins = [i for i, b in enumerate(bins) if b > 0]
        # Remove all positive bins except for the last one
        # Track which ones are removed so we can remove corresponding colors
        if len(pos_bins) > 1:
            bins = np.delete(bins, pos_bins[:-1])
            bin_colors = [color for i, color in enumerate(bin_colors) if i not in pos_bins[:-1]]
        plot_dict['bin_dict']['bins'] = bins

    plot_dict['bin_dict']['n_bins'] = len(bins) - 1
    plot_dict['bin_dict']['bin_colors'] = bin_colors
    cmap = ListedColormap(bin_colors)
    plot_dict['bin_dict']['cmap'] = cmap
    plot_dict['bin_dict']['norm'] = BoundaryNorm(bins, cmap.N, clip = True)