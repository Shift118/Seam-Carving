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

def carve_column(image, cols_to_remove=0):
    """Remove specified number of columns from the image and track seams"""
    # Make a copy of the original image
    current_image = np.array(image)
    rows, cols, channels = current_image.shape
    
    # Store all seams
    all_seams = []
    
    for i in range(cols_to_remove):
        print(f"Removing {i+1}/{cols_to_remove}")
        
        # Convert to grayscale for energy calculation
        gray_image = np.array(Image.fromarray(current_image.astype(np.uint8)).convert('L'))
        
        # Calculate energy map
        energy = apply_sobel(gray_image)
        
        # Find optimal seam
        seam = compute_optimal_seam(energy)
        all_seams.append(seam.copy())
        
        # Remove the seam
        current_image = remove_seam_jit(current_image, seam, rows, cols - i, channels)
    
    return current_image, all_seams

def carve_row(image, rows_to_remove=0):
    """Remove specified number of rows from the image and track seams"""
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image.astype(np.uint8))
    
    # Rotate image 90 degrees
    transposed = pil_image.rotate(90, expand=True)
    
    # Convert back to numpy array
    transposed_array = np.array(transposed)
    
    # Carve columns (which are rows in the original orientation)
    carved_transposed, transposed_seams = carve_column(transposed_array, rows_to_remove)
    
    # Convert back to PIL for rotation
    carved_pil = Image.fromarray(carved_transposed)
    
    # Rotate back
    carved_rotated = carved_pil.rotate(270, expand=True)
    
    # Convert back to numpy array
    carved_array = np.array(carved_rotated)
    
    # Return the carved image and the seams (still in rotated coordinates)
    return carved_array, transposed_seams, transposed_array.shape

def transform_horizontal_seams_to_original(horizontal_seams, transposed_shape, original_shape):
    """Convert horizontal seam coordinates from transposed image back to original image coordinates"""
    rows_transposed, cols_transposed, _ = transposed_shape
    rows_original, cols_original, _ = original_shape
    
    transformed_seams = []
    
    for seam in horizontal_seams:
        transformed_seam = np.zeros_like(seam)
        for i in range(len(seam)):
            # In the transposed image, row = col in original, col = rows - row - 1 in original
            transformed_seam[i, 0] = seam[i, 1]
            transformed_seam[i, 1] = rows_transposed - seam[i, 0] - 1
        transformed_seams.append(transformed_seam)
    
    return transformed_seams

def mark_all_seams(original_image, horizontal_seams, vertical_seams, transposed_shape):
    """Mark all seams on the original image (both horizontal and vertical)"""
    marked_image = original_image.copy()
    rows, cols, _ = original_image.shape
    
    # Mark vertical seams
    for seam in vertical_seams:
        for i in range(len(seam)):
            row, col = seam[i]
            if 0 <= row < rows and 0 <= col < cols:
                marked_image[row, col, 0] = 255  # Red
                marked_image[row, col, 1] = 0    # Green
                marked_image[row, col, 2] = 0    # Blue
    
    # Transform and mark horizontal seams
    transformed_horizontal_seams = transform_horizontal_seams_to_original(
        horizontal_seams, transposed_shape, original_image.shape
    )
    
    for seam in transformed_horizontal_seams:
        for i in range(len(seam)):
            row, col = seam[i]
            if 0 <= row < rows and 0 <= col < cols:
                marked_image[row, col, 0] = 255  # Red
                marked_image[row, col, 1] = 0    # Green
                marked_image[row, col, 2] = 0    # Blue
    
    return marked_image

def carve_seams(image, rows_to_remove=0, cols_to_remove=0):
    """Content-aware resize of an image by removing specified number of rows and columns"""
    # Make copies of the original image
    current_image = np.array(image)
    original_image = np.array(image)
    
    # Create energy image for the original image
    gray_image = np.array(Image.fromarray(original_image).convert('L'))
    energy_image = get_energy_image(gray_image)
    
    horizontal_seams = []
    transposed_shape = None
    
    # Remove rows first (if any)
    if rows_to_remove > 0:
        print(f"Starting row removal...")
        current_image, horizontal_seams, transposed_shape = carve_row(current_image, rows_to_remove)
    
    vertical_seams = []
    
    # Then remove columns (if any)
    if cols_to_remove > 0:
        print(f"Starting column removal...")
        current_image, vertical_seams = carve_column(current_image, cols_to_remove)
    
    # Mark all seams on the original image and energy image
    if rows_to_remove > 0 or cols_to_remove > 0:
        marked_image = mark_all_seams(original_image, horizontal_seams, vertical_seams, transposed_shape)
        marked_energy = mark_all_seams(energy_image, horizontal_seams, vertical_seams, transposed_shape)
    else:
        marked_image = original_image
        marked_energy = energy_image
    
    return current_image, marked_image, marked_energy

def main():
    # Specify your image path here
    image_path = "original.jpeg"  # Replace with your image path
    
    # Specify how many rows and columns to remove
    rows_to_remove = 100  # Adjust as needed
    cols_to_remove = 150  # Adjust as needed

    # Record the start time
    start_time = time.time()  
    
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    print(f"Original dimensions: {image.width}x{image.height}")
    print(f"Removing {rows_to_remove} rows and {cols_to_remove} columns")
    
    # Perform seam carving
    carved_array, marked_array, energy_array = carve_seams(image, rows_to_remove, cols_to_remove)
    
    # Convert back to PIL images
    carved_image = Image.fromarray(carved_array)
    marked_image = Image.fromarray(marked_array)
    energy_image = Image.fromarray(energy_array)
    
    print(f"New dimensions: {carved_image.width}x{carved_image.height}")
    
    # Save outputs
    carved_image.save("app3/carved_image3.jpg")
    marked_image.save("app3/marked_image3.jpg")
    energy_image.save("app3/energy_image3.jpg")
    
    print("Saved images:")
    print("- carved_image.jpg: Image with seams removed")
    print("- marked_image.jpg: Original image with seams marked in red")
    print("- energy_image.jpg: Energy map with seams marked in red")
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Execution time: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()