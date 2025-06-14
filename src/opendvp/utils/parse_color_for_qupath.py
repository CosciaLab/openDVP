import re
from itertools import cycle

import anndata as ad
from matplotlib import colors as mcolors

from opendvp.utils import logger


def parse_color_for_qupath(
    color_dict : dict | None, 
    adata : ad.AnnData, 
    adata_obs_key : str
) -> dict:
    """Parse and convert color definitions to a format compatible with QuPath.

    Parameters
    ----------
    color_dict : dict
        Dictionary mapping category names to color definitions (RGB tuples, lists, or hex strings).
    adata : ad.AnnData
        AnnData object containing observation data.
    adata_obs_key : str
        Key in adata.obs specifying the categorical variable to assign colors to.

    Returns:
    -------
    dict
        Dictionary mapping category names to [R, G, B] lists with values in the range 0-255.
    """
    logger.info("Parsing colors compatible with QuPath")
    
    if color_dict is None: 
        logger.info("No color_dict found, using defaults")
        default_colors = [[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189]]
        color_cycle = cycle(default_colors)
        parsed_colors = dict(
            zip(
                adata.obs[adata_obs_key].cat.categories.astype(str),
                color_cycle,
                strict=False,
            )
        )
        logger.info(f"color_dict created: {parsed_colors}")
    else:
        logger.info("Custom color dictionary passed, adapting to QuPath color format")  
        parsed_colors = {}
        for name, color in color_dict.items():
            if isinstance(color, tuple) and len(color) == 3:
                # Handle RGB fraction tuples (0-1)
                parsed_colors[name] = list(int(c * 255) for c in color)
            elif isinstance(color, list) and len(color) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in color):
                # Already in [R, G, B] format with values 0-255
                parsed_colors[name] = color
            elif isinstance(color, str) and re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color):
                # Handle hex codes
                parsed_colors[name] = mcolors.hex2color(color)
                parsed_colors[name] = list(int(c * 255) for c in parsed_colors[name])
            else:
                raise ValueError(f"Invalid color format for '{name}': {color}")
            
    return parsed_colors