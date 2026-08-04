"""Figure scaffolding.

Verbatim copy from ``idd_forecast_mbp.plot_functions`` (lib/viz
systematization, 2026-06-29). Original retained unchanged.
"""
import cartopy.crs as ccrs  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


def turn_off_axes(axes):
    for ax in axes:
        if ax is not None:
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

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
