import os
import pickle # type: ignore
import numpy as np # type: ignore
import geopandas as gpd # type: ignore
from pathlib import Path # type: ignore
import cartopy.crs as ccrs # type: ignore
import matplotlib.pyplot as plt # type: ignore


from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.xarray_functions import read_netcdf_with_integer_ids 
from idd_forecast_mbp.data_functions import read_polygons

from idd_forecast_mbp.bin_functions import *
from idd_forecast_mbp.color_functions import (
    create_change_colormap, 
    create_outcome_colormap, 
    create_diverging_colors,
    get_colors
)
from idd_forecast_mbp.save_functions import *
from idd_forecast_mbp.data_functions import (load_population_data, 
                                             load_suitability_raster_data, 
                                             load_cov_data, 
                                             get_raster_data, 
                                             get_admin2_data,
                                             update_loc_ids,
                                             get_outcome_df,
                                             get_plot_data)

from idd_forecast_mbp.plot_functions import (plot_data_raster, 
                                             setup_map_plot, 
                                             plot_base_admins, 
                                             plot_data_admins, 
                                             add_inset,
                                             turn_off_axes,
                                             create_figure)
UPLOAD_DATA_PATH = rfc.UPLOAD_DATA_PATH
PROCESSED_DATA_PATH = rfc.PROCESSED_DATA_PATH  # Add this line if not already defined
ssp_scenario_map = rfc.ssp_scenario_map
covariate_map = rfc.covariate_map

best_run_date = "2025_08_11"

model_covariate_ds_path_template = "{UPLOAD_DATA_PATH}/upload_folders/{best_run_date}/cov_ds_{dah_scenario}.nc"
model_covariate_ds_path = model_covariate_ds_path_template.format(UPLOAD_DATA_PATH=UPLOAD_DATA_PATH, best_run_date=best_run_date, dah_scenario='Baseline')
model_covariate_ds = read_netcdf_with_integer_ids(model_covariate_ds_path)
model_covariates = ['urbanization', 'dah_pc', 'flooding_pc', 'gdppc_mean']

bins_dictionary_path = '/ihme/homes/bcreiner/repos/idd-forecast-mbp/notebooks/09_figures/bins_dictionary.pkl'
with open(bins_dictionary_path, 'rb') as f:
    bins_dictionary = pickle.load(f)
admin0_polygons, admin1_polygons, admin2_polygons, disputed_polygons = read_polygons()

def create_map_plot_dict(cause, measure, period_1, ssp_scenarios = ['ssp245'], period_2=None,
                    metric = None, 
                    # Core parameters
                    resolution='0.1', 
                    statistic='mean',
                    map_type='change',
                    per_capita=False,
                    data_type='raster',
                    location_type='endemic', # 'endemic' or 'all'
                    have_legend_panel=True,
                    base_path=None,
                    file_name=None,
                    save_figure=True,
                    remake_figure=False,
                    return_figure=False,
                    display_figure=True,
                    thumbnail=0,
                    existing_fig=None,
                    # Map types:
                    # change: period 1 != period 2; scenario 1 == scenario 2
                    # scenario_comparison: period 1 == period 2; scenario 1 != scenario 2
                    # outcome: period 1 == period 2; scenario 1 == scenario 2
                    # arbitrary_comparison: period 1 != period 2; scenario 1 != scenario 2
                     
                    # Figure information
                    fig_width=12, title_height=0.5, sub_title_height=0, 
                    legend_panel_height=0.75, legend_title_height=0.25,
                    fig_height=8, linewidth=0.05,
                     
                    lat_lon_font_size= 18, inset_label_font_size= 14,
                    tick_font_size=14, water_color='#A6B6DC',
                    water_alpha=0.5,
                    run_date=None,

                    # Bin info
                    zero_bin = False, le = False, ge  = True,
                     # Color infromtation
                    drop_num=0,
                    force_white=True,
                    remove_middle=True,
                    bins = None,
                    abbreviate_labels = False,
                     # Map information
                    add_coasts=False,
                    add_borders=False,
                     # Legend infromation
                     use_colorbar=False,

                     # Map extent
                     lat_min=-60, lat_max=90, lon_min=-180, lon_max=180,
                     lat_zoom_min = -55, lat_zoom_max = 50,
                     
                     # Titles and labels
                     title=None, subtitle=None, subtitle3=None,
                     period_labels=None,legend_title=None,
                     inset_label=None,
                     
                     # Plot styling
                     num_categories=9, custom_bins=None,
                     add_stats=False,
                     
                     # Colors
                     base_cmap=None,
                     masked_color='#f0f0f0', masked_alpha=1.0,
                     
                     # Font sizes
                     title_fontsize=22, legend_title_fontsize=18,
                     legend_label_fontsize=14, stats_fontsize=12,
                     
                     # Layout
                     colorbar_height=0.05, colorbar_pad=0.08, legend_bin_spacing=0.01,
                     legend_margin=0.05, legend_spacing_factor=1.0, 
                     bin_bottom =0.425, bin_top=0.85, bin_label_gap=0.075):

    if map_type == 'outcome':
        periods = [period_1]
    else:
        periods = [period_1, period_2]

    if base_cmap is None:
        if map_type == 'outcome':
            if measure == 'gdppc_mean' or measure == 'dah_pc':
                base_cmap = 'YlGn'
            elif measure == 'suitability' or measure in model_covariates:
                base_cmap = 'Purples'
            elif measure == 'storms':
                base_cmap = 'viridis_r'
            else:
                base_cmap = 'Reds'
        else:
            if measure == 'gdppc_mean' or measure == 'dah_pc':
                base_cmap = 'PiYG'
            elif measure == 'suitability' or measure in model_covariates:
                base_cmap = 'PRGn_r'
            elif measure == 'storms':
                base_cmap = 'viridis_r'
            else:
                base_cmap = 'RdBu_r'
                
    # Use generated subtitle if none provided
    if data_type == 'admin2':
        map_extent = [lon_min, lon_max, lat_zoom_min, lat_zoom_max]
    else:
        map_extent = [lon_min, lon_max, lat_min, lat_max]


    if map_type == 'percent_change':
        prefix_units = ''
        suffix_units = '%'
        le = True
    elif measure == 'gdppc_mean':
        le = True
        prefix_units = '$'
        suffix_units = ''
    else:
        le = le
        prefix_units = ''
        suffix_units = ''

    # Build the plot dictionary
    map_plot_dict = {
        'cause': cause,
        'measure': measure,
        'metric': metric,
        'units': 'people-days' if measure == 'floods' else 'people-hours',
        'map_type': map_type,
        'per_capita': per_capita,
        'data_type': data_type,
        'location_type': location_type,
        'periods': periods,
        'ssp_scenarios': ssp_scenarios,
        'resolution': resolution,
        'statistic': statistic,
        'title': title,
        'subtitle': subtitle,
        'subtitle3': subtitle3,
        'num_categories': num_categories,
        'custom_bins': custom_bins,
        'add_stats': add_stats,
        'have_legend_panel': have_legend_panel,
        'base_path': base_path,
        'file_name': file_name,
        'save_figure': save_figure,
        'display_figure':display_figure,
        'remake_figure': remake_figure,
        'return_figure': return_figure,
        'existing_fig': existing_fig,
        'thumbnail': thumbnail,
        'layout_dict': {
            'fig_width': fig_width,
            'title_height': title_height,
            'sub_title_height': sub_title_height,
            'legend_panel_height': legend_panel_height,
            'legend_title_height': legend_title_height,
        },
        'figure_dict':{
            'linewidth':linewidth,
            'lat_lon_font_size': lat_lon_font_size,
            'inset_label_font_size': inset_label_font_size,
            'tick_font_size': tick_font_size,
            'water_color':water_color,
            'water_alpha':water_alpha,
            'title': title,
            'inset_label': inset_label,
        },
        'bin_dict':{
            'le': le,
            'ge': ge,
            'zero_bin': zero_bin,
            'prefix_units': prefix_units,
            'suffix_units': suffix_units,
            'abbreviate_labels': abbreviate_labels,
        },
        'colors_dict': {
            'base_cmap': base_cmap,
            'water_color': water_color,
            'water_alpha': water_alpha,
            'masked_color': masked_color,
            'masked_alpha': masked_alpha,
            'drop_num': drop_num,
            'force_white': force_white,
            'remove_middle':remove_middle,
        },
        'map_dict': {    
            'admin0_polygons': admin0_polygons,
            'admin1_polygons': admin1_polygons,
            'admin2_polygons': admin2_polygons,
            'plot_admin0s': True,
            'add_coasts': add_coasts,
            'add_borders': add_borders,
            'map_extent': map_extent,
            'raster_extent': [-180, 180, -90, 90]
        },
        'legend_dict': {
            'use_colorbar': use_colorbar,
            'legend_title': legend_title,
            'legend_panel': {
                'legend_bin_spacing': legend_bin_spacing,
                'legend_margin': legend_margin,
                'legend_spacing_factor': legend_spacing_factor,
                'bin_bottom': bin_bottom,
                'bin_top': bin_top,
                'bin_label_gap': bin_label_gap
            },
            'color_bar_dict': {
                'colorbar_height': colorbar_height,
                'colorbar_pad': colorbar_pad,
                'colorbar_width_ratio': 0.7,
                'colorbar_height_ratio': 0.5,
                'shrink': 0.9,
                'pad': 0.15,
                'aspect': 40, 
                'fraction':0.05
            }
        },
        'fontsizes': {
            'title_fontsize': title_fontsize,
            'legend_label_fontsize': legend_label_fontsize,
            'legend_title_fontsize': legend_title_fontsize,
            'stats_fontsize': stats_fontsize,
        }
    }
    map_plot_dict = get_layout_dict(map_plot_dict)
    map_plot_dict = get_period_info(map_plot_dict)
    map_plot_dict['outcome_type'] = f"{measure}_{metric}" if metric else measure
    map_plot_dict['outcome_label'] = f"{measure} {metric}" if metric else measure
    map_plot_dict['full_outcome_label'] = f"{measure} {metric}" if metric else "Suitability (days per year)"
    if legend_title is None:
        map_plot_dict['legend_dict']['legend_title'] = map_plot_dict['full_outcome_label']
        map_plot_dict['replace_titles'] = True
    else: 
        map_plot_dict['replace_titles'] = False

    bin_key = (None, measure, None, map_type) if measure == 'suitability' else (cause, measure, metric, map_type)
    if bins is None:
        if bin_key in bins_dictionary:
            bins = bins_dictionary[bin_key]
        else:
            raise ValueError("Must provide bins if there aren't predefined bins")
    n_bins = len(bins) - 1
    map_plot_dict['bin_dict']['bins'] = bins
    map_plot_dict['bin_dict']['n_bins'] = n_bins
    return map_plot_dict

def plot_map(plot_dict):

    get_save_path(plot_dict)

    if not plot_dict['make_figure']:
        print(f"Figure already exists: {plot_dict['save_path']}")
        return None, None
    # else:
    #     print(f"Creating figure {plot_dict['save_path']}")

    get_labels(plot_dict)
    get_plot_data(plot_dict)
    # plot_dict = calculate_colorbar_params(plot_dict)

    plot_dict['map_plot'] = True
    fig, ax_map, ax_legend = create_figure(plot_dict)
    ax_map = setup_map_plot(ax_map, plot_dict)

    if plot_dict['data_type'] == 'raster':
        plot_data_raster(ax_map, plot_dict)
    else:
        plot_base_admins(ax_map, plot_dict)
        plot_data_admins(ax_map, plot_dict)
    
    disputed_polygons.boundary.plot(ax=ax_map, color='darkgrey', linewidth=0.25,linestyle='--', 
                                 transform=ccrs.PlateCarree())

    figure_dict = plot_dict['figure_dict']
    layout_dict = plot_dict['layout_dict']

    # Set title and labels
    fig.text(0.5, layout_dict['title']['text_y'], figure_dict['title'], ha='center', va='center',
             fontsize=plot_dict['fontsizes']['title_fontsize'])
    #  ax_map.set_title(figure_dict['title'], fontsize=plot_dict['fontsizes']['title_fontsize'])
    # ax_map.set_xlabel("Longitude", fontsize=figure_dict['lat_lon_font_size'])
    # ax_map.set_ylabel("Latitude", fontsize=figure_dict['lat_lon_font_size'])
    
    # Add colorbar and legend
    if plot_dict['have_legend_panel']:
        add_legend(fig, ax_legend, plot_dict)
        if plot_dict['legend_dict']['legend_title'] is not None:
            fig.text(0.5, layout_dict['legend_title']['text_y'], plot_dict['legend_dict']['legend_title'], ha='center', va='center', 
                fontsize=plot_dict['fontsizes']['legend_title_fontsize'])
    else:
        add_legend(fig, ax_map, plot_dict)
    if figure_dict['inset_label'] is not None:
        add_inset(ax_map, figure_dict)


    print(f'Final figure size: {plot_dict['layout_dict']['figsize']}')
    print(f'Map layout coordinates are: {layout_dict['map']['coords']}')
    if plot_dict['have_legend_panel']:
        print(f'Legend coordinates are: {layout_dict['legend']['coords']}')
    if plot_dict['save_figure']:    
        if plot_dict['save_path'] is not None:
            save_figure_as_pdf(fig, plot_dict['save_path'], thumbnail=plot_dict['thumbnail'])
            # save_figure_as_pdf_and_png(fig, plot_dict['save_path'])
        else:
            print("No save path provided or generated, figure not saved.")
    
    if plot_dict['return_figure']:
        if plot_dict['display_figure']:
            return plot_dict, fig
        else:   
            plt.close(fig)
            return plot_dict, fig
        
    else:
        plt.close(fig)  # Add this line to prevent display
        return plot_dict, None


























def get_layout_dict(map_plot_dict):
    layout_dict = map_plot_dict['layout_dict']
    fig_width = layout_dict['fig_width']
    map_extent = map_plot_dict['map_dict']['map_extent']

    aspect_ratio = (map_extent[3] - map_extent[2]) / (map_extent[1] - map_extent[0])

    legend_title_height = layout_dict['legend_title_height']
    legend_panel_height = layout_dict['legend_panel_height']
    map_panel_height = fig_width * aspect_ratio
    sub_title_height = layout_dict['sub_title_height']
    title_height = layout_dict['title_height']
    
    fig_height = title_height + sub_title_height + map_panel_height + legend_panel_height + legend_title_height

    layout_dict['figsize'] = (fig_width, fig_height)
    
    panel_names = ['legend_title', 'legend', 'map', 'sub_title', 'title']
    layout_dict['panel_names'] = panel_names
    heights = [legend_title_height, legend_panel_height, map_panel_height, sub_title_height, title_height]
    height_fractions = [h / fig_height for h in heights]
    for ix, panel in enumerate(panel_names):
        panel_dict = {
            'height': heights[ix],
            'height_fraction': height_fractions[ix],
            'bottom': sum(height_fractions[:ix]),
            'text_y': sum(height_fractions[:ix]) + height_fractions[ix] / 2,
            'coords': [0, sum(height_fractions[:ix]), 1, height_fractions[ix]]
        }
        layout_dict[panel] = panel_dict
    
    keys_to_remove = ['legend_title_height', 'legend_panel_height', 'sub_title_height', 'title_height']
    if not map_plot_dict['have_legend_panel']:
        keys_to_remove.append('legend')
    for key in keys_to_remove:
        del layout_dict[key]
    return map_plot_dict

def get_period_info(map_plot_dict):
    map_type = map_plot_dict['map_type']
    ssp_scenarios = map_plot_dict['ssp_scenarios']
    periods = map_plot_dict['periods']
    def periods_are_different(period):
        if len(period) == 1:
            return False
        else:
            period1, period2 = period
            start1 = period1[0]
            end1 = period1[0] if len(period1) == 1 else period1[1]
            start2 = period2[0] 
            end2 = period2[0] if len(period2) == 1 else period2[1]
            return (start1, end1) != (start2, end2)
    def scenarios_are_different(ssp_scenarios):
        """Check if two scenarios are different."""
        # Normalize scenarios to lower case for comparison
        if len(ssp_scenarios) != 2:
            return False  # Not enough scenarios to compare
        else:
            return ssp_scenarios[0].lower() != ssp_scenarios[1].lower()
    # Initialize period_dict
    if (map_type == 'change') or (map_type == 'percent_change'):
        if len(periods) != 2 or scenarios_are_different(ssp_scenarios):
            raise ValueError("Invalid temporal comparison. Need exactly two periods, 0 or 2 period labels and one SSP scenario.")
        else:
            # Create period configurations for temporal comparison
            for ix, period_years in enumerate(periods):
                if len(period_years) == 1:
                    start_year = end_year = period_years[0]
                else:
                    start_year, end_year = period_years
                map_plot_dict[f'period_{ix+1}'] = {
                    'start_year': start_year,
                    'end_year': end_year,
                    'ssp_scenario': ssp_scenarios[0]
                }
    elif map_type == 'scenario_comparison':
        if len(ssp_scenarios) != 2 or periods_are_different(periods):
            raise ValueError("Invalid scenario comparison. Need exactly two SSP scenarios and one period / period label.")
        else:
            # Create period configurations for scenario comparison (same period, different scenarios)
            period_years = periods[0]
            if len(period_years) == 1:
                start_year = end_year = period_years[0]
            else:
                start_year, end_year = period_years
            
            for ix, ssp_scenario in enumerate(ssp_scenarios):
                map_plot_dict[f'period_{ix+1}'] = {
                    'start_year': start_year,
                    'end_year': end_year,
                    'ssp_scenario': ssp_scenario
                }            
    elif map_type == 'outcome':
        if len(periods) != 1 or len(ssp_scenarios) != 1:
            raise ValueError("For impact evaluation, need exactly one period and one SSP scenario.")
        else:
            # Create single period configuration
            period_years = periods[0]
            if len(period_years) == 1:
                start_year = end_year = period_years[0]
            else:
                start_year, end_year = period_years
            
            map_plot_dict['period_1'] = {
                'start_year': start_year,
                'end_year': end_year,
                'ssp_scenario': ssp_scenarios[0]
            }
            
    elif map_type == 'arbitrary_comparison':
        if len(periods) != 2 or len(ssp_scenarios) != 2:
            raise ValueError("For arbitrary comparison, need exactly two periods and two SSP scenarios.")
        else:
            # Create period configurations for arbitrary comparison
            for ix, (period, ssp_scenario) in enumerate(zip(periods, ssp_scenarios)):
                if len(period) == 1:
                    start_year = end_year = period[0]
                else:
                    start_year, end_year = period   
                map_plot_dict[f'period_{ix+1}'] = {
                    'start_year': start_year,
                    'end_year': end_year,
                    'ssp_scenario': ssp_scenario
                }
            
    else:
        raise ValueError("Invalid plot type. Choose from 'change', 'scenario_comparison', 'outcome', or 'arbitrary_comparison'.")    
    return map_plot_dict

def get_labels(map_plot_dict):
    measure = map_plot_dict['measure']
    if measure == 'suitability':
            map_plot_dict['outcome_type'] = 'suitability'
            map_plot_dict['outcome_label'] = "suitability"
            map_plot_dict['full_outcome_label'] = "suitability (days per year)"
    elif measure in model_covariates:
            map_plot_dict['outcome_type'] = measure
            map_plot_dict['outcome_label'] = covariate_map[measure]['title']
            map_plot_dict['full_outcome_label'] = covariate_map[measure]['ylabel']
    else:
        metric = map_plot_dict['metric']
        map_plot_dict['outcome_type'] = f"{measure}_{metric}"
        map_plot_dict['outcome_label'] = f"{measure} {metric}"
        map_plot_dict['full_outcome_label']  = f"{measure} {metric} (per 100,000 population)" if metric == 'rate' else f"{measure} {metric}"

def get_save_path(map_plot_dict):
    """Generate the save path for the outcome data."""

    base_path = map_plot_dict['base_path']
    file_name = map_plot_dict['file_name']
    Path(base_path).mkdir(parents=True, exist_ok=True)

    map_type = map_plot_dict['map_type']
    outcome_type = map_plot_dict['outcome_type']
    cause = map_plot_dict['cause']

    if file_name is None:
        if cause is None:
                if map_type == 'outcome':
                    save_path = f'{base_path}/{outcome_type}_{map_type}_{map_plot_dict['period_1']['ssp_scenario']}_{map_plot_dict['period_1']['start_year']}'
                elif (map_type == 'change') or (map_type == 'percent_change'):
                    save_path = f'{base_path}/{outcome_type}_{map_type}_{map_plot_dict['period_1']['ssp_scenario']}_{map_plot_dict['period_1']['start_year']}_{map_plot_dict['period_2']['start_year']}'
                else:
                    save_path = f'{base_path}/{outcome_type}_{map_type}_{map_plot_dict['period_1']['ssp_scenario']}_{map_plot_dict['period_2']['ssp_scenario']}_{map_plot_dict['period_1']['start_year']}'
        else:
            if map_type == 'outcome':
                save_path = f'{base_path}/{outcome_type}_{cause}_{map_type}_{map_plot_dict['period_1']['ssp_scenario']}_{map_plot_dict['period_1']['start_year']}'
            elif (map_type == 'change') or (map_type == 'percent_change'):
                save_path = f'{base_path}/{outcome_type}_{cause}_{map_type}_{map_plot_dict['period_1']['ssp_scenario']}_{map_plot_dict['period_1']['start_year']}_{map_plot_dict['period_2']['start_year']}'
            else:
                save_path = f'{base_path}/{outcome_type}_{cause}_{map_type}_{map_plot_dict['period_1']['ssp_scenario']}_{map_plot_dict['period_2']['ssp_scenario']}_{map_plot_dict['period_1']['start_year']}'
        
        if map_plot_dict['statistic'] != 'mean':
            save_path += f'_{map_plot_dict["statistic"]}'
    else:
        save_path = f'{base_path}/{file_name}'


    if not map_plot_dict['remake_figure'] and Path(save_path+'.png').exists():
        map_plot_dict['make_figure'] = False
    else:
        map_plot_dict['make_figure'] = True
    
    map_plot_dict['save_path'] = save_path
