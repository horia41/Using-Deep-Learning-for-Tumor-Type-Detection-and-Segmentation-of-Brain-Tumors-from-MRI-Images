import random
import numpy as np
import h5py
import os
import cv2
import matplotlib.pyplot as plt

# path to dataset
data_dir = "D:\\BACHELOR THESIS\\dataset\\ashkhagan\\figshare-brain-tumor-dataset\\versions\\1\\dataset\\data"
total_images = 3064

X_tumor = []
Y_tumor_mask = []
tumor_labels = []


for i in range(1, total_images + 1):
    filename = str(i) + ".mat"
    file_path = os.path.join(data_dir, filename)

    # mat file using h5py (for matlab v7.3 files)
    with h5py.File(file_path, "r") as data:
        # tumor image, resize and normalize for model input
        image = np.array(data["cjdata"]["image"])
        image = cv2.resize(image, (128, 128))  # Resize to 128x128 pixels
        image = image / 255.0  # Normalize pixel values to [0, 1]
        X_tumor.append(image)

        # tumor mask, resize and threshold into a binary image
        mask = np.array(data["cjdata"]["tumorMask"])
        mask = cv2.resize(mask, (128, 128))
        mask = (mask > 0).astype(np.uint8)  # Convert to binary mask
        Y_tumor_mask.append(mask)

        # tumor label (1: Meningioma, 2: Glioma, 3: Pituitary Tumor)
        label = int(np.array(data["cjdata"]["label"])[0][0])
        tumor_labels.append(label)

# lists to NumPy arrays
# add extra channel dimension (grayscale) for compatibility with most CNNs
X_tumor = np.array(X_tumor).reshape(-1, 128, 128, 1)
Y_tumor_mask = np.array(Y_tumor_mask).reshape(-1, 128, 128, 1)
tumor_labels = np.array(tumor_labels)


print("Dataset Shapes:")
print("X_tumor:", X_tumor.shape)
print("Y_tumor_mask:", Y_tumor_mask.shape)
print("Tumor Labels:", tumor_labels.shape)
print("Unique Labels:", np.unique(tumor_labels))


# visualize

num_samples = 5

sample_indices = random.sample(range(len(X_tumor)), num_samples)

for idx in sample_indices:
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(X_tumor[idx].squeeze(), cmap='gray')
    plt.title(f"Tumor Image\nLabel: {tumor_labels[idx]}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(Y_tumor_mask[idx].squeeze(), cmap='gray')
    plt.title("Tumor Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()