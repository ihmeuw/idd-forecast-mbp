"""Time-series & covariate line panels (single panel + nested grid).

Verbatim copies from ``idd_forecast_mbp.plot_functions`` (lib/viz
systematization, 2026-06-29). Originals retained unchanged.
"""
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.gridspec as gridspec  # type: ignore
from matplotlib.lines import Line2D  # type: ignore

from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.number_functions import get_multiplier
from idd_forecast_mbp.save_functions import save_figure_as_pdf, save_figure_as_png
from idd_forecast_mbp.lib.viz.figures import create_figure
from idd_forecast_mbp.lib.viz.legends import plot_custom_legend

cause_map = rfc.cause_map
ssp_scenario_map = rfc.ssp_scenario_map
full_measure_map = rfc.full_measure_map


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
