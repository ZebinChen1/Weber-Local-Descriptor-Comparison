import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.util import img_as_ubyte
import pandas as pd
import rasterio
from imageio import imwrite
import matplotlib.pyplot as plt
from skimage.io import imread

class local_binary_pattern:
    def __init__(self, img, P, R, method):
        self.img = img
        self.points = P
        self.radius = R
        self.method = method
    def transition_count(self, binary):
        transitions = 0
        for i in range(self.points):
            if binary[i] != binary[(i + 1) % len(binary)]:
                transitions +=1 
            return transitions  
    def algorithm(self):
        h, w = self.img.shape
        lbp_image = np.zeros((h, w), dtype=np.uint8)
        pad = self.radius
        padded = np.pad(self.img, pad, mode='reflect')

        for i in range(pad, h + pad):
            for j in range(pad, w + pad):
                center = padded[i, j]
                code = ''
                for ii in range(-self.radius, self.radius + 1):
                    for jj in range(-self.radius, self.radius + 1):
                        if ii == 0 and jj == 0:
                            continue
                        neighbor = padded[i + ii, j + jj]
                        code += '1' if neighbor >= center else '0'
                if self.method == "default":
                    value = int(code, 2) % 256 if len(code) > 8 else int(code, 2)
                elif self.method == "uniform":
                    count = self.transition_count(code)
                    if count <= 2:
                        value = sum(int(b) for b in code)
                    else:
                        value = self.points + 1  
                else:
                    raise ValueError("Invalid Method: Use Default or Uniform")
                
                lbp_image[i - pad, j - pad] = value
        return lbp_image
def var_img(img, size):
    pad = size //2 
    h, w = img.shape
    var_img = np.zeros((h,w), dtype= np.float32)
    
    for i in range(h-1):
        for j in range(w-1):
            window = img[i:i+size, j:j+size]
            var_img[i,j] = np.var(window)
    return var_img

def get_lbp(img, P=8, R=1, method='default'):
    return local_binary_pattern(img, P, R, method).algorithm()

def get_var(img, size=3):
    return var_img(img, size)

def get_lbp_ri(img, P=8, R=1):
    return local_binary_pattern(img, P, R, method='uniform')
def edge_pad(img, pad):
    H, W = img.shape
    padded = np.zeros((H + 2*pad, W + 2*pad), dtype=img.dtype)

    padded[pad:pad+H, pad:pad+W] = img

    for i in range(H):
        padded[pad+i, :pad] = img[i, 0]      
        padded[pad+i, pad+W:] = img[i, -1]  

    for j in range(W):
        padded[:pad, pad+j] = img[0, j]      
        padded[pad+H:, pad+j] = img[-1, j]  

    padded[:pad, :pad] = img[0, 0]          
    padded[:pad, pad+W:] = img[0, -1]          
    padded[pad+H:, :pad] = img[-1, 0]          
    padded[pad+H:, pad+W:] = img[-1, -1]       

    return padded
def get_wld(img, P=8, R=1):
    img = img.astype(np.float32)
    h, w = img.shape
    excitation_img = np.zeros((h,w), dtype=np.float32)
    orientation_img  = np.zeros((h,w), dtype = np.float32)
    
    angles = np.linspace(0, 2*np.pi, P, endpoint=False)
    dy = -R * np.sin(angles)
    dx =  R * np.cos(angles)

    padded = edge_pad(img, R+1)
    for i in range(h):
        for j in range(w):
            I_c = padded[i+R+1, j+R+1]
            neighbors = []
            for k in range(P):
                y = int(round(i + R + 1 + dy[k]))
                x = int(round(j + R + 1 + dx[k]))
                neighbors.append(padded[y, x])

            excitation_img[i, j] = np.arctan(np.sum((np.array(neighbors) - I_c) / (I_c + 1e-5)))
            grad_y = padded[i+R+2, j+R+1] - padded[i+R, j+R+1]
            grad_x = padded[i+R+1, j+R+2] - padded[i+R+1, j+R]
            orientation_img[i, j] = np.arctan2(grad_y, grad_x)

    excitation_norm = (excitation_img - excitation_img.min()) / (excitation_img.max() - excitation_img.min() + 1e-5) * 255
    orientation_norm = (orientation_img + np.pi) / (2 * np.pi) * 255
    var = get_var(img, size=3)
    var_norm = (var - var.min()) / (var.max() - var.min() + 1e-5) * 255
    #combine into one histogram
    combined = (excitation_norm + orientation_norm + var_norm) / 3
    return combined.astype(np.uint8)


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
### This part is classifying pixel
def classify_pixel(desc_img, class_models, window=40):
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

def run_pipeline(image_path, label_masks, descriptor="lbp", P=8, R=1):
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

