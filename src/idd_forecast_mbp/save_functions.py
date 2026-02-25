import os
import matplotlib.pyplot as plt


from PIL import Image

def save_thumbnail_figure_as_png_1(fig, thumbnail, filename_base, dpi=720, bbox_inches=None):
    """
    Save fig at original size and dpi, then shrink pixel dimensions by 'thumbnail' and save at same dpi.
    """
    if dpi is None or dpi == 'figure':
        dpi = fig.dpi
    print(f"Saving thumbnail at {thumbnail*100:.0f}% size (pixel shrink, keep dpi)")
    # Step 1: Save figure at original size and dpi
    temp_png = f"{filename_base}_temp.png"
    fig.savefig(temp_png, format='png', dpi=dpi, bbox_inches=bbox_inches)
    # Step 2: Read PNG back in
    img = Image.open(temp_png)
    width_px, height_px = img.size
    # Step 3: Shrink to thumbnail pixel dimensions
    new_size_px = (int(width_px * thumbnail), int(height_px * thumbnail))
    img_thumb = img.resize(new_size_px, Image.LANCZOS)
    # Step 4: Save at original dpi
    out_png = f"{filename_base}_thumbnail.png"
    img_thumb.save(out_png, dpi=(dpi, dpi))
    print(f"Thumbnail saved as {out_png} at {new_size_px[0]}x{new_size_px[1]} px and {dpi} dpi.")
    os.remove(temp_png)

def save_thumbnail_figure_as_png_2(fig, thumbnail, filename_base, dpi=360, bbox_inches=None,
                                text_scale=True, line_scale=True):
    if dpi is None or dpi == 'figure':
        dpi = fig.dpi
    print(f"Saving thumbnail at {thumbnail*100:.0f}% size")
    original_figsize = fig.get_size_inches()
    original_fontsize = plt.rcParams['font.size']
    new_fontsize = original_fontsize * thumbnail
    new_linewidth = thumbnail  # You may want to scale this differently

    plt.rcParams.update({'font.size': new_fontsize})
    fig.set_size_inches(original_figsize * thumbnail)

    # Resize all text objects
    if text_scale:
        for ax in fig.get_axes():
            ax.title.set_fontsize(new_fontsize)
            ax.xaxis.label.set_fontsize(new_fontsize)
            ax.yaxis.label.set_fontsize(new_fontsize)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(new_fontsize)
            legend = ax.get_legend()
            if legend:
                legend.get_title().set_fontsize(new_fontsize)
                for text in legend.get_texts():
                    text.set_fontsize(new_fontsize)
            for child in ax.get_children():
                if isinstance(child, plt.Text):
                    child.set_fontsize(new_fontsize)
        for text in fig.texts:
            text.set_fontsize(new_fontsize)

    # Resize all line objects
    if line_scale:
        for ax in fig.get_axes():
            for line in ax.get_lines():
                line.set_linewidth(new_linewidth)
            for child in ax.get_children():
                if hasattr(child, 'set_linewidth'):
                    child.set_linewidth(new_linewidth)

    save_figure_as_png(fig, f"{filename_base}_thumbnail", dpi=dpi, bbox_inches=bbox_inches)
    # Optionally restore original size/font if needed

def save_figure_as_pdf(fig, filename_base, dpi='figure', bbox_inches=None, pad_inches=0, thumbnail=0.0):
    """
    Save a Matplotlib figure as a PDF file.
    Adds '.pdf' to the filename automatically.
    Ensures the directory exists and sets file permissions to 775.
    """
    plt.rcParams['pdf.fonttype'] = 42
    filename = f"{filename_base}.pdf"
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    fig.savefig(filename, format='pdf', dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    os.chmod(filename, 0o775)
    print(f"Figure saved as {filename} (chmod 775)")
    #
    print(thumbnail)
    if thumbnail > 0.0:
        save_thumbnail_figure_as_png_1(fig, thumbnail, filename_base, dpi=dpi, bbox_inches=bbox_inches)
        # save_thumbnail_figure_as_png_2(fig, thumbnail, filename_base, dpi=dpi, bbox_inches=bbox_inches)

def save_figure_as_png(fig, filename_base, dpi=720, bbox_inches=None, pad_inches=0, thumbnail=0.0):
    """
    Save a Matplotlib figure as a PNG file.
    Adds '.png' to the filename automatically.
    Ensures the directory exists and sets file permissions to 775.
    """
    filename = f"{filename_base}.png"
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    fig.savefig(filename, format='png', dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    os.chmod(filename, 0o775)
    print(f"Figure saved as {filename} (chmod 775)")
    #
    if thumbnail > 0.0:
        save_thumbnail_figure_as_png(fig, thumbnail, filename_base, dpi=dpi, bbox_inches=bbox_inches)

def save_figure_as_pdf_and_png(fig, filename_base, pdf_dpi=720, png_dpi=720, bbox_inches=None, pad_inches=0, thumbnail=0.0):
    """
    Save a Matplotlib figure as both PDF and PNG files.
    Adds appropriate extensions automatically.
    Ensures the directory exists and sets file permissions to 775.
    """
    save_figure_as_pdf(fig, filename_base, dpi=pdf_dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, thumbnail=thumbnail)
    save_figure_as_png(fig, filename_base, dpi=png_dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, thumbnail=thumbnail)

