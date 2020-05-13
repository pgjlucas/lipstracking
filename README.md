# Lips Tracking
## Scope
Development of a real time tracking algorithm using computer vision techniques. 
## Tracking architecture
Bottleneck architecture, face to lips.
1. Detect the face from an image and crop the bottom half of the detected bounding box
2. Detect the mouth from the detected bounding box
3. Apply the 2 to the next frame, in the surrounding area of the detected mouth (twice the size)

## Classifier architecture
Our final architecture to detect the mouth from a face is the following:
1. Apply cascade classifier with HOG features and get n bounding boxes:
2. Apply SVM classifier with LBP features on the n bounding boxes and get m bounding boxes
3. Apply classifier with color features on the n bounding boxes and get k bounding boxes
4. For i in k (ordred by confidence):
  - Compute the IoU between the m bounding boxes and i
  - If IoU > Thresh: keep the boxe as detected

## Files

1. Classifiers
   - Classifier1.m : classify using only HOG features
   - Classifier12.m : classify using HOG features and color features.
   - Classifier13.m : classify using HOG features and LBP features.
   - Classifier123.m : classify using HOG, color & LBP features.
   
2. my_detectedMouth.m : Base line of our study using prebuilt functions.

3. train.m : training algorithm for the classifiers with HOG & LBP features.

4. True_False_positive_data.zip: contains 160 images of positive examples and 160 images of negative examples

## Results
- Test set: 25 frames from AVICAR dataset, 4 images per frame (100 images total)
- Faces are correclty detected for the 25 frames 

| Classifier    | Precision |Recall | Inference Time (s) |
| ------------- | -------------- | -------------- | -------------- |
| Classifier 1  |  50%  | 100%	 |    0.013  |
| Classifier 1 + 2  | 99% | 99%	 |    0.027  | 
| Classifier 1 + 3  |58% | 100%  |	  0.038  |
| Whole model	      | 99% | 100% |    0.051  |

The whole model (classifier 1 + 2 + 3) is too slow for the real time tracking. 
Classifier 1 + 2 is good for tracking. 
