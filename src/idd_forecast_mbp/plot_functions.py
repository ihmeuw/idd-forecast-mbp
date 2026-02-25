import numpy as np # type: ignore

import cartopy.crs as ccrs # type: ignore
import cartopy.feature as cfeature # type: ignore

import matplotlib.pyplot as plt # type: ignore
from matplotlib.lines import Line2D # type: ignore
import matplotlib.gridspec as gridspec # type: ignore
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.color_functions import *
from idd_forecast_mbp.bin_functions import *
from idd_forecast_mbp.save_functions import *
from idd_forecast_mbp.number_functions import get_multiplier

cause_map = rfc.cause_map
ssp_scenario_map = rfc.ssp_scenario_map
full_measure_map = rfc.full_measure_map


def create_figure(plot_dict):
    layout_dict = plot_dict['layout_dict']
    existing_fig = plot_dict.get('existing_fig', None)
    map_plot = plot_dict.get('map_plot', None)
    gridplot = plot_dict.get('gridplot', False)
    panel_letter = plot_dict.get('panel_letter', None)

    if existing_fig is not None:
        # Reuse existing figure - clear it completely
        existing_fig.clear()
        fig = existing_fig
        # Make sure the figure size matches (in case it changed)
        fig.set_size_inches(layout_dict['figsize'])
    else:
        # Create new figure (original behavior)
        fig = plt.figure(figsize=layout_dict['figsize'])
    if panel_letter is not None:
        fig.text(0.02, 0.98, panel_letter, fontsize=24, va='top', ha='left', transform=fig.transFigure)
    if map_plot is not None:
        ax_map = fig.add_axes(layout_dict['map']['coords'], projection=ccrs.PlateCarree())
        ax_map.set_position(layout_dict['map']['coords'])
        ax_map.set_aspect('auto')
        ax_legend = fig.add_axes(layout_dict['legend']['coords']) if plot_dict['have_legend_panel'] else None
        turn_off_axes([ax_map, ax_legend])
        return fig, ax_map, ax_legend
    elif gridplot:
        axes = []
        ax_keys = [key for key in layout_dict.keys() if key.startswith('ax')]
        ax_keys.sort()  # Ensure consistent order (ax1, ax2, ax3, etc.)
        for ax_key in ax_keys:
            ax = fig.add_axes(layout_dict[ax_key]['coords'])
            axes.append(ax)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        return fig, *axes
    else:
        ax = fig.add_axes(layout_dict['ax']['coords'])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        return fig, ax

###########################
#### Time series plots ####
###########################

def plot_custom_legend(measures, ssp_scenario_map, legend_labels,ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis('off')

    c_num = len(ssp_scenario_map)
    col_x = [0.25 + 0.18 + 0.18 * i for i in range(c_num)]
    r_num = len(measures)
    row_y = [0.62 - 0.24 * i for i in range(r_num)]

    # Store all text and line objects
    artists = []

    # Column headers
    artists.append(ax.text(0.15, 0.72, 'Weight', ha='right', va='bottom', fontsize=12, fontweight='bold', zorder=1, transform=ax.transAxes))
    artists.append(ax.text(0.25, 0.72, 'Past', ha='center', va='bottom', fontsize=12, zorder=1, transform=ax.transAxes))
    artists.append(ax.text(col_x[1], 0.95, 'RCP', ha='center', va='bottom', fontsize=12, fontweight='bold', zorder=1, transform=ax.transAxes))

    weight_text = ax.text(0.15, 0.72, 'Weight', ha='right', va='bottom', fontsize=12, fontweight='bold', zorder=1, transform=ax.transAxes)
    past_text = ax.text(0.25, 0.72, 'Past', ha='center', va='bottom', fontsize=12, zorder=1, transform=ax.transAxes)
    artists.extend([weight_text, past_text])
    # "Past" lines (black)
    # population (dashed)
    line_past_m0, = ax.plot([0.25 - 0.04, 0.25 + 0.04], [row_y[0]] * 2, color='black', lw=2, linestyle='-', zorder=2, transform=ax.transAxes)
    lines_past = [line_past_m0]
    if len(measures) >= 2:
        line_past_m1, = ax.plot([0.25 - 0.04, 0.25 + 0.04], [row_y[1]] * 2, color='black', lw=2, linestyle='--', zorder=2, transform=ax.transAxes)
        lines_past = [line_past_m0, line_past_m1]
    if len(measures) == 3:
        line_past_m2, = ax.plot([0.25 - 0.04, 0.25 + 0.04], [row_y[2]] * 2, color='black', lw=2, linestyle=':', zorder=2, transform=ax.transAxes)
        lines_past.append(line_past_m2)
    artists.extend(lines_past)

    # RCP numbers
    for i, ssp in enumerate(ssp_scenario_map):
        rcp = ssp_scenario_map[ssp].get('rcp_scenario', '')
        artists.append(ax.text(col_x[i], 0.72, str(rcp), ha='center', va='bottom', fontsize=12, zorder=-2, transform=ax.transAxes))

    # Row headers
    for i, label in enumerate(legend_labels):
        artists.append(ax.text(0.15, row_y[i], label, ha='right', va='center', fontsize=12, zorder=-2, transform=ax.transAxes))

    # Legend lines
    lines = []
    for i, ssp in enumerate(ssp_scenario_map):
        color = ssp_scenario_map[ssp].get('color', 'black')
        line0, = ax.plot([col_x[i] - 0.04, col_x[i] + 0.04], [row_y[0]] * 2, color=color, lw=2, linestyle='-', zorder=-1, transform=ax.transAxes)
        lines = [line0]
        if len(measures) >= 2:
            line1, = ax.plot([col_x[i] - 0.04, col_x[i] + 0.04], [row_y[1]] * 2, color=color, lw=2, linestyle='--', zorder=-1, transform=ax.transAxes)
            lines = [line0, line1]
        if len(measures) == 3:
            line2, = ax.plot([col_x[i] - 0.04, col_x[i] + 0.04], [row_y[2]] * 2, color=color, lw=2, linestyle=':', zorder=-1, transform=ax.transAxes)
            lines.append(line2)
        artists.extend(lines)

    # Draw the canvas to get accurate bounding boxes
    ax.figure.canvas.draw()

    weight_bbox = weight_text.get_window_extent(ax.figure.canvas.get_renderer())
    past_bbox = past_text.get_window_extent(ax.figure.canvas.get_renderer())

    # Find the bounding box of all legend content in display (pixel) coordinates
    bboxes = []
    for artist in artists:
        if hasattr(artist, 'get_window_extent'):
            bbox = artist.get_window_extent(ax.figure.canvas.get_renderer())
            bboxes.append(bbox)
        elif hasattr(artist, 'get_path'):
            # For lines, get the bounding box of the path
            trans = artist.get_transform()
            path = artist.get_path().transformed(trans)
            bbox = path.get_extents()
            bboxes.append(bbox)

    # Combine all bounding boxes
    from matplotlib.transforms import Bbox
    full_bbox = Bbox.union(bboxes)

    # Convert display bbox to axes fraction coordinates
    inv = ax.transAxes.inverted()
    bbox_axes = full_bbox.transformed(inv)
    weight_bbox_axes = weight_bbox.transformed(inv)
    past_bbox_axes = past_bbox.transformed(inv)

    buffer_x = 0.02 * bbox_axes.width
    buffer_y = 0.02 * bbox_axes.height

    rect_x = bbox_axes.x0 - buffer_x
    rect_y = bbox_axes.y0 - buffer_y
    rect_width = bbox_axes.width + 2 * buffer_x
    rect_height = bbox_axes.height + 2 * buffer_y

    # # If you want a white background:
    # ax.add_patch(plt.Rectangle(
    #     (rect_x, rect_y),
    #     rect_width,
    #     rect_height,
    #     color='white',
    #     zorder=-3,
    #     transform=ax.transAxes,
    #     clip_on=False
    # ))



def plot_global_weighted_covariate_single_panel(
    summary_df,
    cause,
    covariate,
    legend_labels,
    ssp_scenario_map,
    covariate_map,
    legend_pos=[0.18, 0.35, 0.35, 0.18],
    figsize=(12, 12*6/10),
    coords = [0.1, 0.1, 0.8, 0.8],
    min_ylim = True,
    max_ylim = False,
    y_grid_gap = 50,
    xlabel='Year',
    panel_letter=None,
    save_pdf=True,save_png=False,
    path = None
    ):

    ylabel = covariate_map[covariate]['ylabel']

    plot_dict={
        'layout_dict':{
            'figsize': figsize,
            'ax': {
                'coords': coords
            }
        },
        'panel_letter': panel_letter,
    }

    fig, ax = create_figure(plot_dict)
    all_handels = []
    rcp_labels = []
    line_styles = ['-', '--', ':']
    multiplier = 1
    if covariate == 'urbanization':
        multiplier = 100

    obs_min = 1e10
    obs_max = -1e10
    for ssp_scenario in ssp_scenario_map:
        handels = []
        scenario_df = summary_df[summary_df['ssp_scenario'] == ssp_scenario]
        after_2023 = scenario_df[scenario_df['year_id'] >= 2025]
        if ssp_scenario == 'ssp245':
            before_2023 = scenario_df[scenario_df['year_id'] <= 2025]
        else:
            before_2023 = None

        color = ssp_scenario_map[ssp_scenario].get('color', None)
        name = ssp_scenario_map[ssp_scenario].get('name', ssp_scenario)
        rcp = ssp_scenario_map[ssp_scenario].get('rcp', '')

        for ix, weight in enumerate(['daly', 'population']):
            if before_2023 is not None:
                y = before_2023[f'{weight}_average']*multiplier
                obs_min = min(obs_min, np.nanmin(y))
                obs_max = max(obs_max, np.nanmax(y))
                ax.plot(
                    before_2023['year_id'],
                    y,
                    color='black',
                    linewidth=2,
                    linestyle=line_styles[ix]
                )
            
            if not after_2023.empty:
                y = after_2023[f'{weight}_average']*multiplier
                obs_min = min(obs_min, np.nanmin(y))
                obs_max = max(obs_max, np.nanmax(y))
                mort_line, = ax.plot(
                    after_2023['year_id'],
                    y,
                    color=color,
                    linewidth=2,
                    linestyle=line_styles[ix]
                )
                handels.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))

        all_handels.append(handels)

        rcp_labels.append(f"{rcp}" if rcp else "")

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.grid(True)

    capitalized_cause = cause.capitalize()
    
    if covariate == 'suitability':
        covariate_title = f'{capitalized_cause} Temperature Suitability'
    else:
        covariate_title = covariate_map[covariate]["title"]
    ax.set_title(f'{capitalized_cause}-weighted {covariate_title}', fontsize=22)


    # Build legend grid
    legend_elements = []
    for i in range(len(ssp_scenario_map)):
        legend_elements.append(all_handels[i])

    # Create a new axes for the legend below the plot
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 1, height_ratios=[8, 1], figure=fig)
    legend_ax = fig.add_axes(legend_pos)
    legend_ax.set_facecolor('white')
    legend_ax.patch.set_alpha(1.0)
    plot_custom_legend(['daly', 'population'], ssp_scenario_map, legend_labels, ax=legend_ax)

    if min_ylim:
        ax.set_ylim(bottom=0)
    if max_ylim:
        ax.set_ylim(top=365)

    ax.set_xlim(left=2000, right=2100)
    import matplotlib.ticker as mticker
    if covariate == 'suitability':
        ax.yaxis.set_major_locator(mticker.MultipleLocator(y_grid_gap))
    if covariate == 'urbanization':
        ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
        ax.set_ylim(top=100)
    if covariate == 'gdppc_mean':
        # Set y axis to log scale
        ax.set_yscale('log')
        custom_ticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000])
        ticks_in_range = custom_ticks[(custom_ticks <= obs_max) & (custom_ticks >= obs_min)]

        lower_tick = custom_ticks[custom_ticks <= obs_min].max() if np.any(custom_ticks <= obs_min) else custom_ticks[0]
        upper_tick = custom_ticks[custom_ticks >= obs_max].min() if np.any(custom_ticks >= obs_max) else custom_ticks[-1]
        final_ticks = np.unique(np.concatenate(([lower_tick], ticks_in_range, [upper_tick])))

        ax.set_yticks(final_ticks)
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    ax.ticklabel_format(style='plain', axis='y')
    ax.tick_params(axis='both', labelsize=14)

    if path is not None:
        if save_pdf:
            save_figure_as_pdf(fig, path, dpi=720, bbox_inches=None)
        if save_png:
            save_figure_as_png(fig, path, dpi=360, bbox_inches=None)
        plt.close(fig)
    else:
        plt.show()


def plot_covariate_single_panel(
    summary_df,
    cause,
    covariate,
    covariate_map,
    figsize=(12, 12*6/10),
    title=None,
    vars = ['Baseline', 'Constant'],
    names = ['Reference', 'Constant'],
    colors = ['blue', 'orange'],
    coords = [0.1, 0.1, 0.8, 0.8],
    min_ylim = True,
    max_ylim = False,
    y_grid_gap = 50,
    xlabel='Year',
    panel_letter=None,
    save_pdf=True,save_png=False,
    path = None
    ):

    ylabel = covariate_map[covariate]['ylabel']

    plot_dict={
        'layout_dict':{
            'figsize': figsize,
            'ax': {
                'coords': coords
            }
        },
        'panel_letter': panel_letter,
    }

    fig, ax = create_figure(plot_dict)
    all_handels = []
    rcp_labels = []
    line_styles = ['-', '--', ':']
    multiplier = 1
    if covariate == 'urbanization':
        multiplier = 100

    obs_min = 1e10
    obs_max = -1e10
    # First, find the max absolute value for scaling
    for ix, var in enumerate(vars):
        var_df = summary_df[summary_df['var'] == var]
        y = var_df['val'] * multiplier
        obs_min = min(obs_min, np.nanmin(y))
        obs_max = max(obs_max, np.nanmax(y))
    max_abs = max(abs(obs_min), abs(obs_max))
    multiplier_auto, multiplier_text = get_multiplier(max_abs)

    # Now plot, scaling by both multipliers
    for ix, var in enumerate(vars):
        var_df = summary_df[summary_df['var'] == var]
        name = names[ix]
        y = var_df['val'] * multiplier * multiplier_auto
        ax.plot(
            var_df['year_id'],
            y,
            color=colors[ix],
            linewidth=2,
            linestyle=line_styles[ix],
            label=name,
            zorder=len(vars) - ix
        )
    # --- Add legend if at least 2 vars ---
    if len(vars) >= 2:
        ax.legend(fontsize=14, loc='best')

    # Update ylabel to include units
    ylabel = covariate_map[covariate]['ylabel'] + multiplier_text
    ax.set_ylabel(ylabel, fontsize=18)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.grid(True)

    if title is not None:
        ax.set_title(title, fontsize=22)
    else:
        capitalized_cause = cause.capitalize()
        if covariate == 'suitability':
            covariate_title = f'{capitalized_cause} Temperature Suitability'
        else:
            covariate_title = covariate_map[covariate]["title"]
        ax.set_title(f'{capitalized_cause} {covariate_title}', fontsize=22)

    if min_ylim:
        ax.set_ylim(bottom=0)
    if max_ylim:
        ax.set_ylim(top=365)

    # Only set plain tick labels if not log scale
    if covariate != 'gdppc_mean':
        ax.ticklabel_format(style='plain', axis='y')

    ax.set_xlim(left=2000, right=2100)
    import matplotlib.ticker as mticker
    if covariate == 'suitability':
        ax.yaxis.set_major_locator(mticker.MultipleLocator(y_grid_gap))
    if covariate == 'urbanization':
        ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
        ax.set_ylim(top=100)
    if covariate == 'gdppc_mean':
        # Set y axis to log scale
        ax.set_yscale('log')
        custom_ticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000])
        ticks_in_range = custom_ticks[(custom_ticks <= obs_max) & (custom_ticks >= obs_min)]
        lower_tick = custom_ticks[custom_ticks <= obs_min].max() if np.any(custom_ticks <= obs_min) else custom_ticks[0]
        upper_tick = custom_ticks[custom_ticks >= obs_max].min() if np.any(custom_ticks >= obs_max) else custom_ticks[-1]
        final_ticks = np.unique(np.concatenate(([lower_tick], ticks_in_range, [upper_tick])))
        ax.set_yticks(final_ticks)
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    ax.tick_params(axis='both', labelsize=14)

    if path is not None:
        if save_pdf:
            save_figure_as_pdf(fig, path, dpi=720, bbox_inches=None)
        if save_png:
            save_figure_as_png(fig, path, dpi=360, bbox_inches=None)
        plt.close(fig)
    else:
        plt.show()

def create_nested_grid_figure(plot_df, plot_info, aa_past_df = None,
                                # Spacing parameters:
                                fig_width=12, fig_height=12*15/16,
                                # Title spacing:
                                title_y=0.9,
                                title_margin=0.05,
                                col_lab_loc = 0.9, row_lab_loc = 0.07,
                                # Panel spacing:
                                hspace_outer=0.25, wspace_outer=0.13,
                                hspace_inner=0.1,
                                legend_height=0.15,
                                panel_left_margin=0.01,
                                panel_right_margin=0.01,
                                # Font options:
                                tick_fontsize=12,
                                label_fontsize=14, legend_fontsize=14,
                                col_label_fontsize=22, row_label_fontsize=22,
                                xlabel_pad=10,
                                path=None):
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Main 2×2 grid (plus 1 row for legend = 3×2)
    main_gs = gridspec.GridSpec(3, 2, figure=fig, 
                               height_ratios=[1, 1, legend_height],  # 2 main rows + legend
                               hspace=hspace_outer, wspace=wspace_outer)
    
    axes_dict = {}
    main_panels_dict = {}
    
    # Define the 2×2 main grid structure:
    main_grid_structure = [
        # Row 0: [Malaria DALY, Malaria Mortality]
        [('malaria', 'daly'), ('malaria', 'mortality')],
        # Row 1: [Dengue DALY, Dengue Mortality]  
        [('dengue', 'daly'), ('dengue', 'mortality')]
    ]

    count_multiplier = []
    count_multiplier_text = []
    for measure in ['daly', 'mortality']:
        sub_df = plot_df[(plot_df['measure'] == measure) & (plot_df['metric'] == 'count') & (plot_df['year_id'] >= plot_info['year_start'])]
        max_val = sub_df['val'].max()
        multiplier, multiplier_text = get_multiplier(max_val, override_multiplier=None)
        count_multiplier.append(multiplier)
        count_multiplier_text.append(multiplier_text)
        
    for main_row in range(2):  # 2 main rows
        for main_col in range(2):  # 2 main columns
            
            cause, measure = main_grid_structure[main_row][main_col]
            main_panel_name = f"{cause}_{measure}_panel"

            # Create 2×1 subgrid within this main grid cell (2 rows, 1 column)
            sub_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1,  # 2 rows, 1 column within each main cell
                main_gs[main_row, main_col],
                hspace=hspace_inner  # Tight spacing between Count and Rate subpanels
            )
            
            main_panels_dict[main_panel_name] = {
                'gridspec': main_gs[main_row, main_col],
                'main_row': main_row,
                'main_col': main_col,
                'bbox': main_gs[main_row, main_col].get_position(fig),
                'cause': cause,
                'measure': measure,
                'sub_axes': []  # Will store the subplot axes
            }

            # Get the cause and measure for this main grid position
            cause, measure = main_grid_structure[main_row][main_col]
            
            sub_panel_dict = {}
            # Create the two subplots (Count and Rate) within this main cell
            for sub_row, metric in enumerate(['count', 'rate']):
                ax = fig.add_subplot(sub_gs[sub_row, 0])
                plot_key = f"{cause}_{measure}_{metric}_{main_row}_{main_col}_{sub_row}"
                axes_dict[plot_key] = ax
                sub_bbox = sub_gs[sub_row, 0].get_position(fig)

                sub_panel_dict[metric] = {
                    'gridspec': sub_gs[sub_row, 0],
                    'sub_row': sub_row,
                    'bbox': sub_bbox,
                }

                main_panels_dict[main_panel_name][f'{metric}_subpanel'] = sub_panel_dict[metric]                
                # Plot the data for this specific combination
                for ssp_scenario in ssp_scenario_map:
                    sub_df = plot_df[
                        (plot_df['location_id'] == plot_info['location_id']) &
                        (plot_df['cause'] == cause) &
                        (plot_df['measure'] == measure) &
                        (plot_df['metric'] == metric) &
                        (plot_df['ssp_scenario'] == ssp_scenario) &
                        (plot_df['year_id'] >= plot_info['year_start'])
                    ]
                    
                    if len(sub_df) > 0:
                        x_values = sub_df['year_id']
                        y_values = sub_df['val']
                        
                        # Apply rate multiplier for rate metrics
                        if metric == 'rate':
                            y_values = y_values * 100000
                        else:
                            y_values = y_values * count_multiplier[main_col]
                        
                        ax.plot(x_values, y_values, 
                               color=ssp_scenario_map[ssp_scenario]['color'],
                               linewidth=2, zorder=3)
                        
                if plot_info['year_start'] < 2022 and aa_past_df is not None: 
                    # 2023 values from ssp scenarios
                    y_last = plot_df[
                        (plot_df['location_id'] == plot_info['location_id']) &
                        (plot_df['cause'] == cause) &
                        (plot_df['measure'] == measure) &
                        (plot_df['metric'] == metric) &
                        (plot_df['ssp_scenario'] == ssp_scenario) &
                        (plot_df['year_id'] == 2022)]['val'].iloc[-1]
                    
                    sub_df = aa_past_df[(aa_past_df['location_id'] == plot_info['location_id']) &
                                        (aa_past_df['cause'] == cause) &
                                        (aa_past_df['measure'] == measure) &
                                        (aa_past_df['metric'] == metric) &
                                        (aa_past_df['year_id'] >= plot_info['year_start'])&
                                        (aa_past_df['year_id'] <= 2022)]
                    if len(sub_df) > 0:
                        x_values = sub_df['year_id']
                        y_values = sub_df['value']

                        y_mult = y_last / y_values.iloc[-1]
                        # y_mult = 1
                        y_values = y_values * y_mult
                        
                        # Apply rate multiplier for rate metrics
                        if metric == 'rate':
                            y_values = y_values * 100000
                        else:
                            y_values = y_values * count_multiplier[main_col]
                        
                        ax.plot(x_values, y_values, 
                               color='black', linewidth=2, zorder=3)
                
                # Formatting
                ax.tick_params(labelsize=tick_fontsize)
                
                # X-axis labels only on bottom subpanel of each main cell
                if sub_row == 1:  
                    ax.set_xlabel('Year', fontsize=label_fontsize, labelpad = xlabel_pad)
                else:
                    ax.set_xticklabels([])
                
                # Y-axis labels on all subpanels
                ylabel = full_measure_map[measure][f'{metric}_name']
                if metric == 'rate':
                    ylabel += '\n (per 100,000)'
                else:
                    ylabel += f'\n{count_multiplier_text[main_col]}'
                # ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad = ylabel_pad)
                ax.text(-.115, 0.5, ylabel, fontsize=label_fontsize, va='center', ha='center', rotation=90, transform=ax.transAxes)

    
    # Create legend in bottom row spanning both columns
    legend_ax = fig.add_subplot(main_gs[2, :])
    if plot_info['year_start'] < 2022: 
        color = ['black'] + [ssp_scenario_map[scenario]['color'] for scenario in ssp_scenario_map]
        label = ['Historical'] + [ssp_scenario_map[scenario]['name'] for scenario in ssp_scenario_map]
        legend_handles = [
            plt.Line2D([], [], color=color[i],
                      label=label[i], linewidth=3)
            for i in range(len(color))
        ]
    else:
        legend_handles = [
            plt.Line2D([], [], color=ssp_scenario_map[scenario]['color'],
                    label=ssp_scenario_map[scenario]['name'], linewidth=3)
            for scenario in ssp_scenario_map
        ]
    legend_ax.legend(handles=legend_handles, loc='center', ncol=len(ssp_scenario_map),
                    fontsize=legend_fontsize, frameon=False)
    legend_ax.axis('off')

    for main_row in range(2):
        cause, measure = main_grid_structure[main_row][0]
        main_panel_name = f"{cause}_{measure}_panel"
        bbox = main_panels_dict[main_panel_name]['bbox']
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(
            row_lab_loc, y_center,  # 0.02 is near the left edge
            cause_map[cause]['cause_name'],   # Replace with your label text
            va='center', ha='center', fontsize=row_label_fontsize, rotation=90
        )
    for main_col in range(2):
        cause, measure = main_grid_structure[0][main_col]
        main_panel_name = f"{cause}_{measure}_panel"
        bbox = main_panels_dict[main_panel_name]['bbox']
        x_center = (bbox.x0 + bbox.x1) / 2
        fig.text(
            x_center, title_y,  # 0.95 is near the top edge
            full_measure_map[measure][f'{plot_info["metrics_to_plot"][0]}_name'],  # Replace with your label text
            va='center', ha='center', fontsize=col_label_fontsize
        )
    
    if plot_info.get('ymin_zero', False):
        for ax in axes_dict.values():
            ax.set_ylim(bottom=0)
            y_max = max([line.get_ydata().max() for line in ax.get_lines()])
            ax.set_ylim(top=y_max * 1.05)

    for ax in axes_dict.values():
        pos = ax.get_position()
        ax.set_position([
            pos.x0 + panel_left_margin, pos.y0,
            pos.width - (panel_left_margin + panel_right_margin), pos.height
        ])
        ax.set_xlim(left=plot_info['year_start'], right=2100)
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        # Optionally, add/subtract a small value for padding
        ax.set_xlim(plot_info['year_start'], xticks[-1])
        ax.set_ylim(yticks[0], yticks[-1])
        ax.grid(True, which='both', linestyle='--', alpha=0.75, zorder=1)

    if path is not None:
        save_figure_as_pdf(fig, path, dpi=720, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        
    return main_panels_dict, axes_dict






###########################
####        Maps       ####
###########################


def plot_data_raster(ax_map, map_plot_dict):
    """Plot the data raster on the map."""
    map_dict = map_plot_dict['map_dict']
    bin_dict = map_plot_dict['bin_dict']
    bins = bin_dict['bins']
    cmap = bin_dict['cmap']
    norm = bin_dict['norm']
    raster_extent = map_dict.get('raster_extent', map_dict.get('map_extent', [-180, 180, -90, 90]))
    
    # Sample colors at bin centers (same as admin2 plotting)
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    actual_bin_colors = []
    
    for center in bin_centers:
        normalized_center = norm(center)
        color = cmap(normalized_center)
        actual_bin_colors.append(color)
    bin_dict['bin_colors'] = actual_bin_colors
    
    # Create colormap with exact colors used
    display_cmap = ListedColormap(actual_bin_colors)
    
    ax_map.imshow(bin_dict['categorical_data'], 
                  cmap=display_cmap, 
                  vmin=0, 
                  vmax=len(actual_bin_colors)-1,
                  transform=ccrs.PlateCarree(), 
                  extent=raster_extent, zorder=2)
    
    if map_dict['plot_admin0s']:
        map_dict['admin0_polygons'].boundary.plot(ax=ax_map, color='darkgrey', linewidth=0.25, 
                                 transform=ccrs.PlateCarree())

def setup_map_plot(ax_map, map_plot_dict):
    map_dict = map_plot_dict['map_dict']
    figure_dict = map_plot_dict['figure_dict']
    layout_dict = map_plot_dict['layout_dict']
    add_coasts = map_dict.get('add_coasts', False)
    add_borders = map_dict.get('add_borders', False)
    add_border = False
    map_extent = map_dict.get('map_extent', [-180, 180, -90, 90])
    ax_map.set_extent(map_extent, crs=ccrs.PlateCarree())

    intended_position = layout_dict['map']['coords']
    ax_map.set_extent(map_extent, crs=ccrs.PlateCarree())
    ax_map.set_position(intended_position)
    ax_map.set_aspect('auto')  # Prevent further adjustments

    # Add geographic features
    ax_map.add_feature(cfeature.OCEAN, facecolor=figure_dict['water_color'], alpha=figure_dict['water_alpha'], zorder=0)
    if add_coasts:
        ax_map.coastlines(linewidth=0.5)
    if add_borders:
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')
    
    return ax_map

def plot_base_admins(ax_map, map_plot_dict):
    """Plot base polygon layers."""
    map_dict = map_plot_dict['map_dict']
    admin0_polygons = map_dict['admin0_polygons']
    admin1_polygons = map_dict['admin1_polygons']
    # Plot all admin2 areas in grey as background
    admin0_polygons.plot(ax=ax_map, color='lightgrey', edgecolor='black', 
                        linewidth=0, transform=ccrs.PlateCarree())
    # if map_dict.get('plot_admin1s', False):
    #     if map_plot_dict['map_a1_loc_ids'] is not None:
    #         admin1s_to_plot = admin1_polygons[~admin1_polygons['location_id'].isin(map_plot_dict['map_a1_loc_ids'])]
    #     else:
    #         admin1s_to_plot = admin1_polygons
    #     admin1s_to_plot.boundary.plot(ax=ax_map, color='darkgrey', linewidth=0.25, 
    #                                         transform=ccrs.PlateCarree())

def plot_data_admins(ax_map, map_plot_dict, linewidth=0):
    """Plot polygons with data colors."""
    map_dict = map_plot_dict['map_dict']
    data_dict = map_plot_dict['data_dict']
    bin_dict = map_plot_dict['bin_dict']
    cmap = bin_dict['cmap']
    norm = bin_dict['norm']
    bins = bin_dict['bins']
    data_column = data_dict['data_column']
    measure = map_plot_dict['measure']

    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    actual_bin_colors = []

    for center in bin_centers:
        normalized_center = norm(center)
        color = cmap(normalized_center)
        actual_bin_colors.append(color)
    bin_dict['bin_colors'] = actual_bin_colors

    if measure != 'dah_pc':
        admin2_endemic = map_dict['admin2_polygons']['location_id'].isin(map_plot_dict['map_a2_loc_ids'])
        # admin2_with_data = admin2_endemic.merge(data_dict['plot_data'], on='location_id', how='left')
        admin2_with_data = map_dict['admin2_polygons'][admin2_endemic].merge(data_dict['plot_data'], on='location_id', how='left')

        admin2_clipped = clip_data_to_bins(admin2_with_data, data_column, bins)
        admin2_clipped.plot(column=data_column, ax=ax_map, cmap=cmap, norm=norm, 
                            legend=False, edgecolor=None, linewidth=linewidth, 
                            transform=ccrs.PlateCarree())
    else:
        plot_data = data_dict['plot_data']
        plot_data = plot_data[['A0_location_id', 'val']].groupby('A0_location_id').max().reset_index()
        plot_data = plot_data.rename(columns={'A0_location_id': 'location_id'})
        plot_data = clip_data_to_bins(plot_data, data_column, bins)
        admin0_with_data = map_dict['admin0_polygons'].merge(plot_data, on='location_id', how='left')
        admin0_with_data.plot(column=data_column, ax=ax_map, cmap=cmap, norm=norm, 
                            legend=False, edgecolor=None, linewidth=linewidth,
                            transform=ccrs.PlateCarree())
    # Add boundaries
    map_dict['admin0_polygons'].boundary.plot(ax=ax_map, color='black', linewidth=0.5, 
                                 transform=ccrs.PlateCarree())
    
def add_inset(ax, figure_dict):
    return
    # """Add legend for non-endemic areas."""
    # inset_elements = [Patch(facecolor='lightgrey', edgecolor='k', label=figure_dict['inset_label'])]
    # ax.legend(handles=inset_elements, loc='lower left', bbox_to_anchor=(0.0, -0.02), 
    #          frameon=False, fontsize=figure_dict['inset_label_font_size'])
    # ax.text(
    #     0.01, 0.01, 
    #     figure_dict['inset_label'],
    #     transform=ax.transAxes,
    #     fontsize=figure_dict['inset_label_font_size'],
    #     va='bottom', ha='left'
    # )

def turn_off_axes(axes):
    for ax in axes:
        if ax is not None:
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")