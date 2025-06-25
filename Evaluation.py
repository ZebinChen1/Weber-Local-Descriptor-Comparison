from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.util import img_as_ubyte
from train_test import evaluate_all_scales


img = imread("crop.png", as_gray=True)
img = img_as_ubyte(img)


mask_building = np.zeros_like(img, dtype=np.uint8)
mask_building[0:200, 200:750] = 1  

mask_road = np.zeros_like(img, dtype=np.uint8)
mask_road[350:680, 550:850] = 1  

mask_vegetation = np.zeros_like(img, dtype=np.uint8)
mask_vegetation[530:750, 150:450] = 1 

mask_test_building = np.zeros_like(img, dtype=np.uint8)
mask_test_building[0:200, 750:900] = 1  

mask_test_road = np.zeros_like(img, dtype=np.uint8)
mask_test_road[350:680, 850:1000] = 1  

mask_test_vegetation = np.zeros_like(img, dtype=np.uint8)
mask_test_vegetation[0:300, 0:100] = 1 

masks = {
    "building": mask_building,
    "road": mask_road,
    "vegetation": mask_vegetation
}
testing_masks = {
    "building" : mask_test_building,
    "road" : mask_test_road,
    "vegetation" : mask_test_vegetation
}

overlay = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
overlay[mask_test_building > 0] = [255, 0, 0]      
overlay[mask_test_road > 0] = [0, 255, 0]        
overlay[mask_test_vegetation > 0] = [0, 0, 255]    
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap='gray')
ax.imshow(overlay, alpha=0.4) 


legend_elements = [
    Patch(facecolor='red', edgecolor='r', label='Building'),
    Patch(facecolor='green', edgecolor='g', label='Road'),
    Patch(facecolor='blue', edgecolor='b', label='Vegetation')
]
ax.legend(handles=legend_elements, loc='lower right')
ax.axis('off')
plt.show()

df = evaluate_all_scales("crop.png", masks, testing_masks)
