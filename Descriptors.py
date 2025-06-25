import numpy as np
class local_binary_pattern:
    """
    Computes Local Binary Pattern (LBP) descriptors for a grayscale image.

    Attributes:
        img (ndarray): Input grayscale image.
        points (int): Number of circularly symmetric neighbor points for descriptors
        radius (int): Radius parameter for descriptors
        method (str): Either 'default' or 'uniform'
    """
    def __init__(self, img, P, R, method):
        """
        Initialize the LBP class with image and parameters.

        Args:
            img (ndarray): Grayscale input image.
            P (int): Number of circularly symmetric neighbor points for descriptors
            R (int):  Radius parameter for descriptors
            method (str): Type of LBP ('default' or 'uniform').
        """
        self.img = img
        self.points = P
        self.radius = R
        self.method = method
    def transition_count(self, binary):
        """
        Count number of 0-1 or 1-0 transitions in a binary string.

        Args:
            binary (str): Binary code string.

        Returns:
            transitions : Number of transitions in the binary pattern.
        """
        transitions = 0
        for i in range(self.points):
            if binary[i] != binary[(i + 1) % len(binary)]:
                transitions +=1 
            return transitions  
    def algorithm(self):
        """
        Compute LBP descriptor for the entire image.

        Returns:
            lbp_image: LBP-coded image of same size as input.
        """
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
    """
    Compute local variance image using a sliding window.

    Args:
        img (ndarray): Input grayscale image.
        size (int): Window size.

    Returns:
        var_img : Image of local variance values.
    """

    pad = size //2 
    h, w = img.shape
    var_img = np.zeros((h,w), dtype= np.float32)
    
    for i in range(h-1):
        for j in range(w-1):
            window = img[i:i+size, j:j+size]
            var_img[i,j] = np.var(window)
    return var_img
#Get functions
def get_lbp(img, P=8, R=1, method='default'):
    return local_binary_pattern(img, P, R, method).algorithm()

def get_var(img, size=3):
    return var_img(img, size)

def get_lbp_ri(img, P=8, R=1):
    return local_binary_pattern(img, P, R, method='uniform')

def edge_pad(img, pad):
    """
    Apply edge replication padding.

    Args:
        img (ndarray): Input grayscale image.
        pad (int): Padding width.

    Returns:
        padded: Edge-padded image.
    """
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
    """
    Compute Weber Local Descriptor (WLD) combining excitation, orientation, and local variance.

    Args:
        img (ndarray): Input grayscale image.
        P (int): Number of neighbors for directional analysis for descriptor
        R (int): Radius for neighborhood sampling for descriptor

    Returns:
        combined: Combined WLD image scaled to [0, 255].
    """
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

