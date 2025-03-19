"""
Basic Usage Example
==================

This example demonstrates basic usage of PhenoScope for image processing.
"""

# %%
# First, let's import the necessary modules
import numpy as np
import matplotlib.pyplot as plt
import phenoscope as ps

# %%
# Create a simple test image
test_image = np.zeros((100, 100, 3), dtype=np.uint8)
test_image[25:75, 25:75, 0] = 255  # Red square
test_image[40:60, 40:60, 1] = 255  # Green square inside

# %%
# Create an Image object from the numpy array
img = ps.Image(test_image, input_schema='RGB')
print(f"Image shape: {img.shape}")
print(f"Image schema: {img.schema}")

# %%
# Display the image
plt.figure(figsize=(8, 8))
# Convert to numpy array for display
img_array = np.array(img.array[:])
plt.imshow(img_array)
plt.title("Test Image")
plt.axis('off')
plt.show()

# %%
# Extract features from the image
# For demonstration, we'll just get some basic statistics
mean_values = np.mean(img_array[:], axis=(0, 1))
print(f"Mean RGB values: R={mean_values[0]:.1f}, G={mean_values[1]:.1f}, B={mean_values[2]:.1f}")

# %%
# Create a mask for the red square
red_mask = img_array[:, :, 0] > 200
plt.figure(figsize=(8, 8))
plt.imshow(red_mask, cmap='gray')
plt.title("Red Square Mask")
plt.axis('off')
plt.show()

# %%
# Count the number of pixels in the red square
red_pixel_count = np.sum(red_mask)
print(f"Number of red pixels: {red_pixel_count}")
print(f"Expected number of pixels: {50 * 50}")
