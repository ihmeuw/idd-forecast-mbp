"""Map orchestrators: build the map_plot_dict and render a map.

Copied from ``idd_forecast_mbp.map_functions`` (lib/viz systematization,
2026-06-29) and refactored so the data that used to be module-level globals
(admin polygons, disputed polygons, the bins dictionary) is passed in as
explicit arguments. Importing this module therefore has NO side effects and
NO hardcoded paths. The original map_functions.py is retained unchanged.

Typical use:
    from idd_forecast_mbp.data_functions import read_polygons
    a0, a1, a2, disputed = read_polygons()
    d = create_map_plot_dict(..., admin0_polygons=a0, admin1_polygons=a1,
                             admin2_polygons=a2, bins_dictionary=bins_dict)
    plot_map(d, disputed_polygons=disputed)
"""
import cartopy.crs as ccrs  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from idd_forecast_mbp.save_functions import save_figure_as_pdf, save_figure_as_png
# NOTE: add_legend (bin_functions) and get_plot_data (data_functions) are imported
# LAZILY inside plot_map() — those legacy modules currently fail at import (they
# reference the removed constants.PROCESSED_DATA_PATH and load lsae_1209-era files at
# import time). Deferring keeps `import idd_forecast_mbp.lib.viz.map_plot` working and
# create_map_plot_dict usable; plot_map itself still needs that layer to run.
from idd_forecast_mbp.lib.viz.figures import create_figure
from idd_forecast_mbp.lib.viz.maps import (
    plot_data_raster, setup_map_plot, plot_base_admins, plot_data_admins, add_inset,
)
from idd_forecast_mbp.lib.viz.map_layout import (
    get_layout_dict, get_period_info, get_labels, get_save_path, model_covariates,
)


def create_map_plot_dict(cause, measure, period_1, ssp_scenarios = ['ssp245'], 
                         dah_scenarios = ['Baseline'] * 2,
                        hold_variables = [None] * 2,
                        period_2=None,
                    metric = None, 
                    # Core parameters
                    resolution='0.1', 
                    statistic='mean',
                    map_type='change',
                    per_capita=False,
                    data_type='raster',
                    extent='zoom',
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
                    save_pdf=True,save_png=False,
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
                     bin_bottom =0.425, bin_top=0.85, bin_label_gap=0.075,
                     # --- injected data (were module-level globals in map_functions.py) ---
                     admin0_polygons=None, admin1_polygons=None,
                     admin2_polygons=None, bins_dictionary=None):

    if admin0_polygons is None or admin1_polygons is None or admin2_polygons is None:
        raise ValueError(
            "create_map_plot_dict needs admin0_polygons / admin1_polygons / "
            "admin2_polygons passed in (e.g. from "
            "idd_forecast_mbp.data_functions.read_polygons()).")

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
    if extent == 'global':
        map_extent = [lon_min, lon_max, lat_min, lat_max]
    else:
        map_extent = [lon_min, lon_max, lat_zoom_min, lat_zoom_max]

    if map_type == 'percent_change':
        prefix_units = ''
        suffix_units = '%'
        if measure != 'dah_pc':
            le = True
    elif measure == 'gdppc_mean' or measure == 'dah_pc':
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
        'dah_scenarios':dah_scenarios,
        'hold_variables': hold_variables,
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
        'save_pdf': save_pdf,
        'save_png': save_png,
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
        if bins_dictionary is not None and bin_key in bins_dictionary:
            bins = bins_dictionary[bin_key]
        else:
            raise ValueError("Must provide bins (or a bins_dictionary containing this "
                             "key) if there aren't predefined bins")
    n_bins = len(bins) - 1
    map_plot_dict['bin_dict']['bins'] = bins
    map_plot_dict['bin_dict']['n_bins'] = n_bins
    return map_plot_dict


def plot_map(plot_dict, disputed_polygons):
    # Deferred imports (see module header): only triggered when plot_map is actually
    # called, so importing this module doesn't pull in the currently-broken legacy layer.
    from idd_forecast_mbp.bin_functions import add_legend
    from idd_forecast_mbp.data_functions import get_plot_data

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
    
    print(f"Figure size: {fig.get_size_inches()}")
    print(f"ax_map position: {ax_map.get_position().bounds}")
    if ax_legend is not None:
        print(f"ax_legend position: {ax_legend.get_position().bounds}")

    from shapely.geometry import box

    # Create a bounding box from your map extent
    map_extent = plot_dict['map_dict']['map_extent']
    bbox = box(map_extent[0], map_extent[2], map_extent[1], map_extent[3])
    disputed_clipped = disputed_polygons.clip(bbox)
    disputed_clipped.boundary.plot(
        ax=ax_map, 
        color='darkgrey', 
        linewidth=0.25,
        linestyle='--', 
        transform=ccrs.PlateCarree(),
        aspect='auto'  # This might help
    )
    print(f"ax_map xlim: {ax_map.get_xlim()}")
    print(f"ax_map ylim: {ax_map.get_ylim()}")
    print(f"Map extent setting: {map_extent}")

    print(f"Figure size: {fig.get_size_inches()}")
    print(f"ax_map position: {ax_map.get_position().bounds}")
    if ax_legend is not None:
        print(f"ax_legend position: {ax_legend.get_position().bounds}")

    figure_dict = plot_dict['figure_dict']
    layout_dict = plot_dict['layout_dict']
    ax_map.set_position(layout_dict['map']['coords'])
    ax_map.set_aspect('auto')

    map_extent = plot_dict['map_dict']['map_extent']
    ax_map.set_xlim(map_extent[0], map_extent[1])
    ax_map.set_ylim(map_extent[2], map_extent[3])

    # Set title and labels
    fig.text(0.5, layout_dict['title']['text_y'], figure_dict['title'], ha='center', va='center',
             fontsize=plot_dict['fontsizes']['title_fontsize'])
    #  ax_map.set_title(figure_dict['title'], fontsize=plot_dict['fontsizes']['title_fontsize'])
    # ax_map.set_xlabel("Longitude", fontsize=figure_dict['lat_lon_font_size'])
    # ax_map.set_ylabel("Latitude", fontsize=figure_dict['lat_lon_font_size'])
    print(f"Figure size: {fig.get_size_inches()}")
    print(f"ax_map position: {ax_map.get_position().bounds}")
    if ax_legend is not None:
        print(f"ax_legend position: {ax_legend.get_position().bounds}")
    
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

    print(f"Figure size: {fig.get_size_inches()}")
    print(f"ax_map position: {ax_map.get_position().bounds}")
    if ax_legend is not None:
        print(f"ax_legend position: {ax_legend.get_position().bounds}")

    print(f'Final figure size: {plot_dict['layout_dict']['figsize']}')
    print(f'Map layout coordinates are: {layout_dict['map']['coords']}')
    if plot_dict['have_legend_panel']:
        print(f'Legend coordinates are: {layout_dict['legend']['coords']}')
    if plot_dict['save_figure']:    
        if plot_dict['save_path'] is not None:
            if plot_dict['save_pdf']:
                save_figure_as_pdf(fig, plot_dict['save_path'], thumbnail=plot_dict['thumbnail'])
            if plot_dict['save_png']:
                save_figure_as_png(fig, plot_dict['save_path'])
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
