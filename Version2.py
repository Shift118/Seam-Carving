import numpy as np
from scipy.signal import convolve2d
from PIL import Image
from numba import jit
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

@jit(nopython=True)
def is_in_image(row, col, rows, cols):
    """Check if a position is within image boundaries"""
    return 0 <= row < rows and 0 <= col < cols

@jit(nopython=True)
def compute_optimal_seam(energy):
    """Compute optimal seam using dynamic programming with JIT"""
    rows, cols = energy.shape
    
    # Create cost matrix
    dp = energy.copy()
    
    # Create matrix to store the direction of the next pixel in the seam
    backtrack = np.zeros_like(dp, dtype=np.int32)
    
    # Fill dynamic programming table from second row to last
    for row in range(1, rows):
        for col in range(cols):
            # Handle the left edge
            if col == 0:
                if dp[row-1, col] <= dp[row-1, col+1]:
                    backtrack[row, col] = 1  # straight up
                    dp[row, col] = energy[row, col] + dp[row-1, col]
                else:
                    backtrack[row, col] = 2  # up-right
                    dp[row, col] = energy[row, col] + dp[row-1, col+1]
            # Handle the right edge
            elif col == cols-1:
                if dp[row-1, col-1] <= dp[row-1, col]:
                    backtrack[row, col] = 0  # up-left
                    dp[row, col] = energy[row, col] + dp[row-1, col-1]
                else:
                    backtrack[row, col] = 1  # straight up
                    dp[row, col] = energy[row, col] + dp[row-1, col]
            # Handle the middle
            else:
                if dp[row-1, col-1] <= dp[row-1, col] and dp[row-1, col-1] <= dp[row-1, col+1]:
                    backtrack[row, col] = 0  # up-left
                    dp[row, col] = energy[row, col] + dp[row-1, col-1]
                elif dp[row-1, col] <= dp[row-1, col-1] and dp[row-1, col] <= dp[row-1, col+1]:
                    backtrack[row, col] = 1  # straight up
                    dp[row, col] = energy[row, col] + dp[row-1, col]
                else:
                    backtrack[row, col] = 2  # up-right
                    dp[row, col] = energy[row, col] + dp[row-1, col+1]
    
    # Find the index of the minimum value in the last row
    min_val = dp[rows-1, 0]
    seam_end_col = 0
    for col in range(1, cols):
        if dp[rows-1, col] < min_val:
            min_val = dp[rows-1, col]
            seam_end_col = col
    
    # Backtrack to find the seam
    seam = np.zeros((rows, 2), dtype=np.int32)
    seam[rows-1, 0] = rows-1
    seam[rows-1, 1] = seam_end_col
    
    for row in range(rows-1, 0, -1):
        col = seam[row, 1]
        direction = backtrack[row, col]
        
        if direction == 0:  # up-left
            seam[row-1, 0] = row-1
            seam[row-1, 1] = col-1
        elif direction == 1:  # straight up
            seam[row-1, 0] = row-1
            seam[row-1, 1] = col
        else:  # up-right
            seam[row-1, 0] = row-1
            seam[row-1, 1] = col+1
    
    return seam

@jit(nopython=True)
def remove_seam_jit(image, seam, rows, cols, channels):
    """Remove a seam from an image with JIT optimization"""
    output = np.zeros((rows, cols - 1, channels), dtype=np.uint8)
    
    for i in range(rows):
        row = seam[i, 0]
        col = seam[i, 1]
        
        # Copy pixels from the left of the seam
        for c in range(channels):
            for j in range(col):
                output[row, j, c] = image[row, j, c]
        
        # Copy pixels from the right of the seam
        for c in range(channels):
            for j in range(col, cols - 1):
                output[row, j, c] = image[row, j + 1, c]
    
    return output

@jit(nopython=True)
def mark_seam_jit(image, seam, rows):
    """Mark a seam in red with JIT optimization"""
    marked = image.copy()
    
    for i in range(rows):
        row = seam[i, 0]
        col = seam[i, 1]
        marked[row, col, 0] = 255  # Red
        marked[row, col, 1] = 0    # Green
        marked[row, col, 2] = 0    # Blue
    
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
    
    # Get dimensions
    rows, cols, channels = current_image.shape
    
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
        all_seams.append(seam.copy())
        
        # Remove the seam
        current_image = remove_seam_jit(current_image, seam, rows, cols - i, channels)
    
    # Mark all seams on the original image
    marked_image = original_image.copy()
    marked_energy = energy_image.copy()
    
    # Mark each seam
    for seam in all_seams:
        marked_image = mark_seam_jit(marked_image, seam, rows)
        marked_energy = mark_seam_jit(marked_energy, seam, rows)
    
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
    carved_image = Image.fromarray(carved_array)
    marked_image = Image.fromarray(marked_array)
    energy_image = Image.fromarray(energy_array)
    
    print(f"New dimensions: {carved_image.width}x{carved_image.height}")
    
    # Save outputs
    carved_image.save("app2/carved_image_app2.jpg")
    marked_image.save("app2/marked_image2.jpg")
    energy_image.save("app2/energy_image2.jpg")
    
    print("Saved images:")
    print("- carved_image.jpg: Image with seams removed")
    print("- marked_image.jpg: Original image with seams marked in red")
    print("- energy_image.jpg: Energy map with seams marked in red")
    
    end_time = time.time()  # Record the end time

    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Execution time: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()