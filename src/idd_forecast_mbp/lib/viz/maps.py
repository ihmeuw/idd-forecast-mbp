"""Map drawing primitives (raster + admin polygons).

Verbatim copies from ``idd_forecast_mbp.plot_functions`` (lib/viz
systematization, 2026-06-29). Original retained unchanged.
"""
import cartopy.crs as ccrs  # type: ignore
import cartopy.feature as cfeature  # type: ignore
from matplotlib.colors import ListedColormap  # type: ignore

from idd_forecast_mbp.bin_functions import clip_data_to_bins


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
