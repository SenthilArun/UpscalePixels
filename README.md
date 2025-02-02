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
