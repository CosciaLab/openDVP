def lazy_image_check(image_path):
    """Check the image metadata without loading the image"""
    logger.info(" ---- lazy_image_check : version number 1.0.0 ----")
    time_start = time.time()

    with tifffile.TiffFile(image_path) as image:
        # Getting the metadata
        shape = image.series[0].shape
        dtype = image.pages[0].dtype

        n_elements = np.prod(shape)
        bytes_per_element = dtype.itemsize
        estimated_size_bytes = n_elements * bytes_per_element
        estimated_size_gb = estimated_size_bytes / 1024 / 1024 / 1024 
        
        logger.info(f"Image shape is {shape}")
        logger.info(f"Image data type: {dtype}")
        logger.info(f"Estimated size: {estimated_size_gb:.4g} GB")

    logger.info(f" ---- lazy_image_check is done, took {int(time.time() - time_start)}s  ----")