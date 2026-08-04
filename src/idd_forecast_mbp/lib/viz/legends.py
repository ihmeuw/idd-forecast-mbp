"""Custom time-series legend.

Verbatim copy from ``idd_forecast_mbp.plot_functions`` (lib/viz
systematization, 2026-06-29). Original retained unchanged.

(``plot_custom_legend`` imports matplotlib locally, so this module needs no
top-level matplotlib import.)
"""


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
