__all__ = [
    "draw_legend_bins",
    "add_legend",
    "get_bin_info",
    "smart_format",
    "pretty_bin_labels",
    "add_colorbar",
    "add_legend_panel_colorbar",
    "clip_data_to_bins"
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
from idd_forecast_mbp.color_functions import (
    create_change_colormap, 
    create_outcome_colormap, 
    create_diverging_colors,
    get_colors
)


def clip_data_to_bins(df, data_column, bins):
    """Clip data values to ensure they fall within bin boundaries."""
    df = df.copy()
    min_bin, max_bin = bins[0], bins[-1]
    
    # Show clipping stats
    original_min, original_max = df[data_column].min(), df[data_column].max()
    clipped_count = ((df[data_column] < min_bin) | (df[data_column] > max_bin)).sum()
    
    if clipped_count > 0:
        print(f"Clipping {clipped_count} values outside bins [{min_bin}, {max_bin}]")
        print(f"Original range: [{original_min:.2f}, {original_max:.2f}]")
    
    # Clip the data
    df[data_column] = df[data_column].clip(lower=min_bin, upper=max_bin)
    
    return df

def draw_legend_bins(ax, map_plot_dict):
    map_type = map_plot_dict['map_type']
    legend_dict = map_plot_dict['legend_dict']
    bin_dict = map_plot_dict['bin_dict']
    legend_panel = legend_dict['legend_panel']
    legend_bin_spacing = legend_panel['legend_bin_spacing']
    legend_margin = legend_panel['legend_margin']
    bin_colors = bin_dict['bin_colors']
    
    bin_bottom = legend_panel['bin_bottom']
    bin_top = legend_panel['bin_top']
    bin_label_gap = legend_panel['bin_label_gap']

    bin_height = bin_top - bin_bottom
    bin_label_y = bin_bottom - bin_label_gap
    
    category_labels = map_plot_dict['bin_dict']['bin_labels']
    
    bin_dict['n_bins'] = len(category_labels)
    n_bins = bin_dict['n_bins']

    bin_width = (1 - 2 * legend_margin - (n_bins - 1) * legend_bin_spacing) / n_bins
    
    # bin_width = min(bin_width, 0.1)
    bin_left = np.arange(legend_margin, legend_margin + n_bins * (bin_width + legend_bin_spacing), bin_width + legend_bin_spacing)
    bin_center = bin_left + bin_width / 2
    bin_shift = 0.5 - bin_center.mean()
    bin_left += bin_shift
    # bin_left = np.arange(margin, margin + bin_dict['n_bins'] * (bin_width + legend_bin_spacing), bin_width + legend_bin_spacing)

    # print("bin_left:", bin_left)
    # Draw legend rectangles and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    for i in range(n_bins):
        rect = Rectangle((bin_left[i], bin_bottom), bin_width, bin_height, facecolor=bin_colors[i], edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(bin_left[i] + bin_width / 2, bin_label_y, category_labels[i], ha='center', va='top',
                        fontsize=map_plot_dict['fontsizes']['legend_label_fontsize'])

def add_legend(fig, ax, map_plot_dict):
    if map_plot_dict['have_legend_panel']:
        if map_plot_dict['legend_dict']['use_colorbar']:
            add_legend_panel_colorbar(fig, ax, map_plot_dict)
        else:
            draw_legend_bins(ax, map_plot_dict)
    else:
        add_colorbar(fig, ax, map_plot_dict)

def get_bin_info(map_plot_dict, plot_data):
    map_type = map_plot_dict['map_type']

    if map_type == 'outcome':
        # Outcome map
        create_outcome_colormap(map_plot_dict)            
    else:
        create_change_colormap(map_plot_dict)

    map_plot_dict['bin_dict']['bin_labels'] = pretty_bin_labels(map_plot_dict)
    bin_dict = map_plot_dict['bin_dict']
    bins = bin_dict['bins']
    n_bins = bin_dict['n_bins']
    plot_data_values = plot_data.values if hasattr(plot_data, 'values') else plot_data
    categorical_data = np.full_like(plot_data_values, np.nan)
    for i in range(n_bins):
        if i == 0.0:
            mask = plot_data_values <= bins[i+1]
        elif i == n_bins - 1:
            mask = plot_data_values > bins[i]
        else:
            mask = (plot_data_values > bins[i]) & (plot_data_values <= bins[i+1])
        categorical_data[mask] = i

    bin_dict['categorical_data'] = categorical_data

def smart_format(val):
    # If integer, format as int with commas. If <1, keep up to 2 decimals, else keep up to 2 decimals but drop trailing zeros.
    if float(val).is_integer():
        return f"{int(val):,}"
    else:
        s = f"{val:,.2f}".rstrip('0').rstrip('.')
        return s
    
def pretty_bin_labels(map_plot_dict):
    bins = map_plot_dict['bin_dict']['bins']
    le = map_plot_dict['bin_dict']['le']
    ge = map_plot_dict['bin_dict']['ge']
    zero_bin = map_plot_dict['bin_dict']['zero_bin']
    prefix_units = map_plot_dict['bin_dict']['prefix_units']
    suffix_units = map_plot_dict['bin_dict']['suffix_units']
    abbreviate_labels = map_plot_dict['bin_dict']['abbreviate_labels']
    # bins: array-like of bin edges
    # fmt: format for numbers (default: 2 significant digits)

    bins = bins.copy()
    if min(bins) < 0:
        gap = ' to '
    else:
        gap = '–'

    if zero_bin:
        zero_ix = np.where(np.atleast_1d(bins) == 0)[0]

    if prefix_units is not None:
        abbreviate_labels = True

    if abbreviate_labels:
        for ix, bin in enumerate(bins):
            if abs(bin) >= 1_000_000:
                bins[ix] = f'{prefix_units}{smart_format(bin/1000000)}M'
            elif abs(bin) >= 1_000:

                bins[ix] = f'{prefix_units}{smart_format(bin/1000)}K'
            else:
                bins[ix] = f'{prefix_units}{smart_format(bin)}'
    elif suffix_units is not None:
        for ix, bin in enumerate(bins):
            bins[ix] = f'{smart_format(bin)}{suffix_units}'
    else:
        for ix, bin in enumerate(bins):
            bins[ix] = smart_format(bin)

    labels = []
    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i+1]
        if zero_bin:
            # Only check zero_ix if zero_bin is truthy
            if i == zero_ix:
                labels.append(f'0')
            elif i == zero_ix + 1:
                labels.append(f"0 - {right}")
            elif left == right:
                labels.append(left)
            else:
                labels.append(f"{left}{gap}{right}")
        else:
            if left == right:
                labels.append(left)
            else:
                labels.append(f"{left}{gap}{right}")
    if le:
        labels[0] = f"< {bins[1]}"
    if ge:
        labels[-1] = f"> {bins[-2]}"

    if prefix_units == '$':
        labels = [rf"{label.replace('$', r'\$')}" for label in labels]

    return labels

def add_colorbar(fig, ax, map_plot_dict):
    figure_dict = map_plot_dict['figure_dict']
    legend_dict = map_plot_dict['legend_dict']
    bins = map_plot_dict['bin_dict']['bins']
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    sm = plt.cm.ScalarMappable(cmap=map_plot_dict['bin_dict']['cmap'], norm=map_plot_dict['bin_dict']['norm'])
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', 
                        shrink=legend_dict['color_bar_dict']['shrink'], 
                        pad=legend_dict['color_bar_dict']['pad'],
                        aspect=legend_dict['color_bar_dict']['aspect'],
                        fraction=legend_dict['color_bar_dict']['fraction'],
                        ticks=bin_centers)
    cbar.set_ticklabels(figure_dict['bin_labels'], fontsize=figure_dict['tick_font_size'])
    cbar.set_label(figure_dict['colorbar_label'], fontsize=figure_dict['colorbar_title_font_size'])

def add_legend_panel_colorbar(fig, ax_legend, map_plot_dict):
    """
    Fixed version that uses the calculated parameters properly
    """
    ax_legend.clear()
    ax_legend.axis('off')

    # Get colorbar data
    bin_dict = map_plot_dict['bin_dict']
    cmap = bin_dict['cmap'] 
    norm = bin_dict['norm']
    bins = bin_dict.get('bins', None)
    bin_labels = bin_dict.get('bin_labels', None)
    colorbar_label = map_plot_dict['full_outcome_label']
    
    color_bar_dict = map_plot_dict['legend_dict']['color_bar_dict']

    # Create ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])


    exact_fit = True
    legend_pos = ax_legend.get_position()
    # Use the ratios from calculate_colorbar_params
    colorbar_width_ratio = color_bar_dict.get('colorbar_width_ratio', 0.7)
    colorbar_height_ratio = color_bar_dict.get('colorbar_height_ratio', 0.3)

            
    # Calculate colorbar size and position
    cbar_width = legend_pos.width * colorbar_width_ratio
    cbar_height = legend_pos.height * colorbar_height_ratio
    
    cbar_left = legend_pos.x0 + cbar_width
    cbar_bottom = legend_pos.y0 + cbar_height

    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

    extend_option = 'max' if map_plot_dict.get('extend_colorbar', False) else 'neither'
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend=extend_option)

    if bins is not None and bin_labels is not None:
        if hasattr(norm, 'boundaries'):
            # For BoundaryNorm, use the boundaries
            boundaries = list(norm.boundaries)
            # Remove infinite values for display
            display_boundaries = [b for b in boundaries if not np.isinf(b)]
            if len(display_boundaries) > len(bin_labels):
                display_boundaries = display_boundaries[:len(bin_labels)]
            
            cbar.set_ticks(display_boundaries)
            cbar.set_ticklabels(bin_labels[:len(display_boundaries)])
        else:
            # Fallback for other norm types
            tick_positions = np.linspace(norm.vmin, norm.vmax, len(bin_labels))
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(bin_labels)
        
        # Style the colorbar
        cbar.set_label(colorbar_label, fontsize=10, labelpad=5)
        cbar.ax.tick_params(labelsize=9, rotation=0, pad=2)

    return cbar