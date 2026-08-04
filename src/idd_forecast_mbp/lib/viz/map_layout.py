"""Map config / layout / label / save-path helpers (pure dict logic).

Verbatim copies from ``idd_forecast_mbp.map_functions`` (lib/viz
systematization, 2026-06-29). Original retained unchanged.

NOTE: the orchestrators ``create_map_plot_dict()`` and ``plot_map()`` were
deliberately NOT copied here. They rely on import-time data loading
(polygons, a pickled bins dictionary at a hardcoded path) and an external
``add_legend()``; they need a data-injection refactor before they belong in
a library module. They remain in ``map_functions.py`` for now.
"""
from pathlib import Path

from idd_forecast_mbp import constants as rfc

covariate_map = rfc.covariate_map
# Covariates treated as model inputs (was a module global in map_functions.py).
model_covariates = ['urbanization', 'dah_pc', 'flooding_pc', 'gdppc_mean']


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
        if panel == 'title':
            weight_y = 0.4
        elif panel == 'legend_title':
            weight_y = 0.7
        else:
            weight_y = 0.5
        panel_dict = {
            'height': heights[ix],
            'height_fraction': height_fractions[ix],
            'bottom': sum(height_fractions[:ix]),
            'text_y': sum(height_fractions[:ix]) + height_fractions[ix] * weight_y,
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
    dah_scenarios = map_plot_dict.get('dah_scenarios', ['Baseline'] * len(periods))
    hold_variables = map_plot_dict.get('hold_variables', [None] * len(periods))
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
            for ix, (period_years, dah_scenario, hold_variable) in enumerate(zip(periods, dah_scenarios, hold_variables)):
                if len(period_years) == 1:
                    start_year = end_year = period_years[0]
                else:
                    start_year, end_year = period_years
                map_plot_dict[f'period_{ix+1}'] = {
                    'start_year': start_year,
                    'end_year': end_year,
                    'ssp_scenario': ssp_scenarios[0],
                    'dah_scenario': dah_scenario,
                    'hold_variable': hold_variable
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

            for ix, (ssp_scenario, dah_scenario, hold_variable) in enumerate(zip(ssp_scenarios, dah_scenarios, hold_variables)):
                map_plot_dict[f'period_{ix+1}'] = {
                    'start_year': start_year,
                    'end_year': end_year,
                    'ssp_scenario': ssp_scenario,
                    'dah_scenario': dah_scenario,
                    'hold_variable': hold_variable
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
            for ix, (period, ssp_scenario, dah_scenario, hold_variable) in enumerate(zip(periods, ssp_scenarios, dah_scenarios, hold_variables)):
                if len(period) == 1:
                    start_year = end_year = period[0]
                else:
                    start_year, end_year = period   
                map_plot_dict[f'period_{ix+1}'] = {
                    'start_year': start_year,
                    'end_year': end_year,
                    'ssp_scenario': ssp_scenario,
                    'dah_scenario': dah_scenario,
                    'hold_variable': hold_variable
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
