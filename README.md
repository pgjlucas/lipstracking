# Lips Tracking
## Scope
Development of a real time lips tracking algorithm using computer vision techniques. 
## Architecture
Bottlneck architecture
1. Detect the face from an image and crop the bottom half of the detected bounding box
2. Detect the mouth from the detected bounding box
3. Apply the 2 to the next frame, in the surrounding area of the detected mouth (twice the size)

## Results

