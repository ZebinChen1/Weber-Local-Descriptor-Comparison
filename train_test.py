import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.util import img_as_ubyte
import pandas as pd
from Descriptors import get_lbp, get_wld
from utils import patch_histogram, classify_pixel

def run_pipeline(image_path, label_masks, descriptor="lbp", P=8, R=1):

    """
    Full classification pipeline for a grayscale image using a given texture descriptor.
    Generates a descriptor image, trains class histograms using label masks,
    and classifies each pixel.

    Args:
        image_path (str): Path to grayscale image
        label_masks (dict): Dictionary mapping masks obtained from gt labeling
        descriptor (str): Type of descriptor to use
        P (int): Number of neighbors 
        R (int): Radius of neighborhood 

    Returns:
        classified: Image of classified labels, with integer values for each class.
    """
    
    img = io.imread(image_path, as_gray=True)
    img = img_as_ubyte(img)

    if descriptor == "lbp":
        desc_img = get_lbp(img, P, R, method='default')
    elif descriptor == "lbpriu":
        desc_img = get_lbp(img, P, R, method='uniform')
    elif descriptor == "wld":
        desc_img = get_wld(img, P, R)
    else:
        raise ValueError("Invalid descriptor")

    class_models = []
    for ___, mask in label_masks.items():
        hist = patch_histogram(desc_img, mask)
        class_models.append(hist)

    classified = classify_pixel(desc_img, class_models)

    plt.imshow(classified, cmap='tab10')
    plt.title(f"Classified Image - {descriptor.upper()}")
    plt.colorbar()
    plt.show()

    return classified

def compute_accuracy(pred, testing_masks):
    """
    Compute classification accuracy by comparing predictions with ground truth masks.

    Args:
        pred (ndarray): Predicted label image from the classifier
        testing_masks (dict): Dictionary of ground truth masks for each class

    Returns:
        acc : Accuracy in percentage.
    """
    all_gt = []
    all_pred = []

    label_map = {name: i for i, name in enumerate(testing_masks.keys())}

    for name, mask in testing_masks.items():
        true_label = label_map[name]
        indices = np.where(mask > 0)
        all_gt.extend([true_label] * len(indices[0]))
        all_pred.extend(pred[indices])

    all_gt = np.array(all_gt)
    all_pred = np.array(all_pred)

    correct = np.sum(all_gt == all_pred)
    total = len(all_gt)
    acc = correct / total * 100
    return acc

def evaluate_all_scales(image_path, label_masks, test_masks):
    """
    Evaluate classification accuracy across multiple scales (P,R) 

    Args:
        image_path (str): Path to the input image.
        label_masks (dict): Dictionary of binary masks used for training class histograms.
        test_masks (dict): Dictionary of binary masks used for testing accuracy.

    Returns:
        DataFrame: Pandas DataFrame containing classification accuracy for each method and scale.
    """
    
    results = []
    for (P, R) in [(8,1), (16,2), (24,3)]:
        acc_lbp  = compute_accuracy(run_pipeline(image_path, label_masks, "lbp", P, R), label_masks, test_masks)
        acc_ri   = compute_accuracy(run_pipeline(image_path, label_masks, "lbpriu", P, R), label_masks, test_masks)
        acc_wld  = compute_accuracy(run_pipeline(image_path, label_masks, "wld", P, R), label_masks, test_masks)
        results.append([f"{P},{R}", acc_lbp, acc_ri, acc_wld])

    df = pd.DataFrame(results, columns=["P,R", "LBP (%)", "LBPRIU (%)", "WLD (%)"])
    df.loc[len(df)] = ["Mean"] + df.iloc[:, 1:].mean().tolist()
    print(df.to_string(index=False))
    return df

