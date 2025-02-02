import cv2
import cupy as cp
import numpy as np
import time

# Load the image
image_path = "kids.jpg"  # Change this to your image file
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

# Convert image to RGB (OpenCV loads as BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Set upscale factor for more pixels
scale_factor = 4  # Change this value for 2x, 4x, 8x upscale

# Get original dimensions
height, width, channels = image.shape
new_height, new_width = height * scale_factor, width * scale_factor

# Print pixel counts
input_pixels = height * width
upscaled_pixels = new_height * new_width
print(f"Input Image: {width}x{height} ({input_pixels} pixels)")
print(f"Upscaled Image: {new_width}x{new_height} ({upscaled_pixels} pixels)")

# Convert image to GPU array
image_gpu = cp.asarray(image, dtype=cp.uint8)

# **✅ Corrected Triple-Quoted String for CUDA Kernel**
bilinear_kernel = cp.RawKernel(r'''
extern "C" __global__
void upscale_bilinear(const unsigned char* src, unsigned char* dst, int w, int h, int nw, int nh, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Prevent memory access errors
    if (x >= nw || y >= nh || x < 0 || y < 0) return;

    float x_ratio = float(w - 1) / (nw - 1);
    float y_ratio = float(h - 1) / (nh - 1);

    int x_l = min(max(int(x_ratio * x), 0), w - 1);
    int y_l = min(max(int(y_ratio * y), 0), h - 1);
    int x_h = min(x_l + 1, w - 1);
    int y_h = min(y_l + 1, h - 1);

    float x_weight = (x_ratio * x) - x_l;
    float y_weight = (y_ratio * y) - y_l;

    for (int c = 0; c < channels; c++) {
        int tl = src[(y_l * w + x_l) * channels + c];
        int tr = src[(y_l * w + x_h) * channels + c];
        int bl = src[(y_h * w + x_l) * channels + c];
        int br = src[(y_h * w + x_h) * channels + c];

        float top = tl * (1 - x_weight) + tr * x_weight;
        float bottom = bl * (1 - x_weight) + br * x_weight;
        dst[(y * nw + x) * channels + c] = (unsigned char)(top * (1 - y_weight) + bottom * y_weight);
    }
}
''', 'upscale_bilinear')  # ✅ Properly Closed String

# Allocate memory for output image on GPU
output_gpu = cp.zeros((new_height, new_width, channels), dtype=cp.uint8)

# CUDA Grid and Block size (optimized for memory efficiency)
block_size = (32, 32)
grid_size = ((new_width + block_size[0] - 1) // block_size[0],
             (new_height + block_size[1] - 1) // block_size[1])

# **Benchmark CUDA Execution Time**
start_time = time.time()
bilinear_kernel(grid_size, block_size, (image_gpu, output_gpu, width, height, new_width, new_height, channels))
cp.cuda.runtime.deviceSynchronize()  # Ensure kernel execution is complete
cuda_time = (time.time() - start_time) * 1000  # Convert to milliseconds
print(f"CUDA Upscaling Execution Time: {cuda_time:.2f} ms")

# Copy back to CPU
upscaled_image = cp.asnumpy(output_gpu)

# Convert back to BGR for OpenCV
upscaled_image = cv2.cvtColor(upscaled_image, cv2.COLOR_RGB2BGR)

# Save full-sized upscaled image
cv2.imwrite("upscaled_image.jpg", upscaled_image)

# Get Screen Size
screen_width = 1280  # Adjust this based on your screen size
screen_height = 720   # Adjust this based on your screen size

# Resize for Display Only (Preserving Aspect Ratio)
aspect_ratio = new_width / new_height
if new_width > screen_width or new_height > screen_height:
    if aspect_ratio > 1:  # Wider than tall
        display_width = screen_width
        display_height = int(screen_width / aspect_ratio)
    else:  # Taller than wide
        display_height = screen_height
        display_width = int(screen_height * aspect_ratio)
else:
    display_width, display_height = new_width, new_height

# Resize the image for display
display_image = cv2.resize(upscaled_image, (display_width, display_height), interpolation=cv2.INTER_AREA)

# Show the resized image
cv2.imshow("Upscaled Image (Resized to Fit Screen)", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
