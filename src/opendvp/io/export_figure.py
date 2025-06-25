import os
import time

import matplotlib
import matplotlib.figure

from opendvp.utils import logger


def export_figure(
    fig : matplotlib.figure.Figure, 
    path_to_dir : str,
    suffix : str, 
    dpi :int = 300
) -> None:
    """Save a matplotlib figure as both PDF and SVG files with a timestamped filename.

    The function creates the output directory if it does not exist, generates a
    timestamped filename using the current date and time, and saves the figure in
    both PDF and SVG formats with the specified DPI. For SVG export, it ensures
    that text remains editable in vector graphics editors like Adobe Illustrator.

    Parameters:
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure object to be saved.
    path_to_dir : str
        Directory path where the figure files will be saved.
    suffix : str
        A custom suffix to include in the filename (e.g., describing the content).
    dpi : int, optional
        Resolution in dots per inch for the saved figure. Default is 300.

    Returns:
    -------
    None

    Prints:
    ------
    str
        Confirmation message with the full paths to the saved PDF and SVG files.

    Examples:
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 1], [0, 1])
    >>> export_figure(fig, path="figures/", suffix="line_plot")

    Figures saved as: 
    figures/20250519_1245_line_plot.pdf
    figures/20250519_1245_line_plot.svg
    """
    os.makedirs(path_to_dir, exist_ok=True)

    # Ensure editable text in SVG
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rc('font', family='arial')
    matplotlib.rcParams['svg.fonttype'] = 'none'
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    datetime_str = time.strftime("%Y%m%d_%H%M")
    base_filename = f"{datetime_str}_{suffix}"

    pdf_path = os.path.join(path_to_dir, f"{base_filename}.pdf")
    svg_path = os.path.join(path_to_dir, f"{base_filename}.svg")

    fig.savefig(fname=pdf_path, format="pdf", dpi=dpi, bbox_inches="tight", transparent=True)
    fig.savefig(fname=svg_path, format="svg", dpi=dpi, bbox_inches="tight", transparent=True)

    logger.info(f"Figure saved as: {pdf_path} and {svg_path}")