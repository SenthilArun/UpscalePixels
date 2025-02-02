# UpscalePixels
PixelsUpscale
Overview of the Upscaling Process
When upscaling an image, we take a low-resolution (LR) image and generate a high-resolution (HR) image.
This is done using bilinear interpolation, a method that smoothly increases image resolution by estimating new pixel values from nearby pixels.

🛠 Steps of the Code
Load the image using OpenCV (cv2).
Convert the image to a CuPy (GPU) array.
Define a CUDA kernel for bilinear interpolation.
Allocate memory for the upscaled image on the GPU.
Launch the CUDA kernel to process each pixel in parallel.
Copy the upscaled image back from the GPU to the CPU.
Save and display the output image.


LeetCode Problem	Relevance to Bilinear Interpolation
[LC 566] Reshape the Matrix	Grid transformation & mapping pixels
[LC 835] Image Overlap	Weighted sum of neighboring values
[LC 48] Rotate Image	Remapping pixels in a transformation
[LC 289] Game of Life	Neighbor-based calculations in a 2D grid
[LC 36] Valid Sudoku	Accessing structured grid neighbors
[LC 42] Trapping Rain Water	Spatial interpolation in a matrix
[LC 296] Best Meeting Point	Finding intermediate values
