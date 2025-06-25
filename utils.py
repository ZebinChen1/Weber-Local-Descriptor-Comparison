import numpy as np

def bhattacharyya(h1, h2):
    """
    Bhattacharya distance comparing differences between two histograms
    
    Args:
    h1: Trained Histogram
    h2: Ground Truth Histogram

    Returns:
        Computed distance
    """
    return -np.log(np.sum(np.sqrt(h1 * h2)) + 1e-10)

def patch_histogram(desc_img, mask, bins=256):
    """
    computes a normalized histogram of a descriptor image (WLD, WLD_Var, etc)
    Args:
        desc_img: our grayscaled image
        mask: masked region
        bins: # of bins in histogram

    Returns:
        hist: noramlized histogram
    """
    #acts as boolean mask selecting the active pixels
    values = desc_img[mask > 0]
    #forms the histogram with the given values ranging from 0 to the number of bins we want
    hist, _ = np.histogram(values, bins=bins, range=(0, bins), density=True)
    return hist
### Classify pixel

def classify_pixel(desc_img, class_models, window=40):
    """
    Classify each pixel in a descriptor image by comparing the local histogram
    of a patch around the pixel with a list of precomputed class model histograms.

    Args:
        desc_img (ndarray): Descriptor image (ex: LBP encoded)
        class_models (list): List of class histograms representing texture classes
        window (int): Size of the square patch used for histogram computation. 

    Returns:
        result : Image of same height and width as `desc_img`, where each pixel contains the predicted class label.
    """
    h, w = desc_img.shape
    result = np.zeros((h, w), dtype=int)
    pad = window // 2
    padded = np.pad(desc_img, pad, mode='reflect')

    for i in range(h):
        for j in range(w):
            patch = padded[i:i+window, j:j+window]
            hist, _ = np.histogram(patch, bins=256, range=(0, 256), density=True)
            dists = [bhattacharyya(hist, model) for model in class_models]
            result[i, j] = np.argmin(dists)
    return result