import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import time

# Sobel kernels for edge detection
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobel_y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

# Gaussian kernel for blur
gaussian_kernel = (1/16) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

def apply_sobel(image):
    """Apply Sobel operator to compute energy map of an image"""
    # Apply Gaussian blur to reduce noise
    blurred = convolve2d(image, gaussian_kernel, mode='same', boundary='symm')
    
    # Calculate gradients
    grad_x = convolve2d(blurred, sobel_x, mode='same', boundary='symm')
    grad_y = convolve2d(blurred, sobel_y, mode='same', boundary='symm')
    
    # Compute magnitude of gradients
    energy = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    
    # Normalize to 0-255 range
    energy = energy * (255.0 / np.max(energy))
    
    return energy

def compute_optimal_seam(energy):
    """Compute optimal seam using dynamic programming"""
    rows, cols = energy.shape
    
    # Create cost matrix
    dp = energy.copy()
    
    # Create matrix to store the direction of the next pixel in the seam
    backtrack = np.zeros_like(dp, dtype=int)
    
    # Fill dynamic programming table from second row to last
    for row in range(1, rows):
        for col in range(cols):
            # Handle the left edge
            if col == 0:
                idx = np.argmin([dp[row-1, col], dp[row-1, col+1]])
                backtrack[row, col] = idx + 1  # 1=straight, 2=right
                dp[row, col] = energy[row, col] + dp[row-1, col+idx]
            # Handle the right edge
            elif col == cols-1:
                idx = np.argmin([dp[row-1, col-1], dp[row-1, col]])
                backtrack[row, col] = idx  # 0=left, 1=straight
                dp[row, col] = energy[row, col] + dp[row-1, col-1+idx]
            # Handle the middle
            else:
                idx = np.argmin([dp[row-1, col-1], dp[row-1, col], dp[row-1, col+1]])
                backtrack[row, col] = idx  # 0=left, 1=straight, 2=right
                dp[row, col] = energy[row, col] + dp[row-1, col-1+idx]
    
    # Find the index of the minimum value in the last row
    seam_end_col = np.argmin(dp[-1])
    
    # Backtrack to find the seam
    seam = []
    col = seam_end_col
    
    for row in range(rows-1, -1, -1):
        seam.append((row, col))
        if row > 0:  # Stop at the first row
            direction = backtrack[row, col]
            if direction == 0:  # Left
                col = col - 1
            elif direction == 2:  # Right
                col = col + 1
            # If direction == 1, we go straight up (col unchanged)
    
    # Reverse the seam to go from top to bottom
    seam.reverse()
    
    return seam

def remove_seam(image, seam):
    """Remove a seam from an image"""
    rows, cols, channels = image.shape
    output = np.zeros((rows, cols - 1, channels), dtype=image.dtype)
    
    for row, col in seam:
        # Copy pixels from the left of the seam
        output[row, :col] = image[row, :col]
        # Copy pixels from the right of the seam
        output[row, col:] = image[row, col + 1:]
    
    return output

def mark_seam(image, seam):
    """Mark a seam in red"""
    marked = image.copy()
    for row, col in seam:
        marked[row, col] = [255, 0, 0]  # Red color
    return marked

def get_energy_image(gray_image):
    """Compute energy map and convert to RGB image"""
    energy = apply_sobel(gray_image)
    # Convert energy map to RGB
    energy_rgb = np.stack([energy, energy, energy], axis=-1).astype(np.uint8)
    return energy_rgb

def carve_seams(image, cols_to_remove=0):
    """Remove specified number of columns from the image"""
    # Make a copy of the original image
    current_image = np.array(image)
    original_image = np.array(image)
    
    # Create energy image
    gray_image = np.array(Image.fromarray(current_image.astype(np.uint8)).convert('L'))
    energy_image = get_energy_image(gray_image)
    
    all_seams = []
    
    for i in range(cols_to_remove):
        print(f"Removing column {i+1}/{cols_to_remove}")
        
        # Convert to grayscale for energy calculation
        gray_image = np.array(Image.fromarray(current_image.astype(np.uint8)).convert('L'))
        
        # Calculate energy map
        energy = apply_sobel(gray_image)
        
        # Find optimal seam
        seam = compute_optimal_seam(energy)
        all_seams.append(seam)
        
        # Remove the seam
        current_image = remove_seam(current_image, seam)
    
    # Mark all seams on the original image
    marked_image = original_image.copy()
    marked_energy = energy_image.copy()
    
    # Adjust seams for previously removed columns
    for i, seam in enumerate(all_seams):
        # Mark current seam
        for row, col in seam:
            if 0 <= row < marked_image.shape[0] and 0 <= col < marked_image.shape[1]:
                marked_image[row, col] = [255, 0, 0]  # Red color
                marked_energy[row, col] = [255, 0, 0]  # Red color
        
        # Adjust future seams to account for removed columns
        for future_seam in all_seams[i+1:]:
            for j, (row, col) in enumerate(future_seam):
                # If this pixel is to the right of a removed seam, shift it left
                if col >= seam[row][1]:
                    future_seam[j] = (row, col - 1)
    
    return current_image, marked_image, marked_energy

def main():
    # Specify your image path here
    image_path = "original.jpeg"  # Replace with your image path
    
    # Specify how many columns to remove
    cols_to_remove = 267  # Adjust as needed
    
    # Record the start time
    start_time = time.time()  

    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    print(f"Original dimensions: {image.width}x{image.height}")
    print(f"Removing {cols_to_remove} columns")
    
    # Perform seam carving
    carved_array, marked_array, energy_array = carve_seams(image, cols_to_remove)
    
    # Convert back to PIL images
    carved_image = Image.fromarray(carved_array.astype(np.uint8))
    marked_image = Image.fromarray(marked_array.astype(np.uint8))
    energy_image = Image.fromarray(energy_array.astype(np.uint8))
    
    print(f"New dimensions: {carved_image.width}x{carved_image.height}")
    
    # Save outputs
    carved_image.save("app1/carved_image_app1.jpg")
    marked_image.save("app1/marked_image_app1.jpg")
    energy_image.save("app1/energy_image_app1.jpg")
    
    print("Saved images:")
    print("- carved_image.jpg: Image with seams removed")
    print("- marked_image.jpg: Original image with seams marked in red")
    print("- energy_image.jpg: Energy map with seams marked in red")
    end_time = time.time()  # Record the end time

    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Execution time: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()