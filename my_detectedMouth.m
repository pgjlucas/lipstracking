function IMouth=my_detectedMouth(image,downSamplingRatio, thresholdMouth)

% Baseline Face: first, detect the faces
faceDetector = vision.CascadeObjectDetector;
Im = rgb2gray(image) ;
I = Im(1:downSamplingRatio:end,1:downSamplingRatio:end);
I_rezised = image(1:downSamplingRatio:end,1:downSamplingRatio:end);
bboxes = faceDetector(I);  
% We have many false positive, eyes being labelled as mouth. In order to
% correct that, we decided to detect eyes and apply an IoU. 

bboxes_m1 =[] ;
bboxes_e1 = [] ;
bboxes_m = [] ;
bboxes_e = [] ;
for i = 1:size(bboxes,1)
    x_s = bboxes(i,1) ; y_s = bboxes(i,2) ; x_e = x_s + bboxes(i,3) ; y_e = y_s + bboxes(i,4) ;
    face = I(y_s:y_e,x_s:x_e);
    mouthDetector = vision.CascadeObjectDetector('Mouth','MergeThreshold',thresholdMouth) ;
    eyesDetector = vision.CascadeObjectDetector('EyePairBig') ;
    bboxes_m1 = mouthDetector(face);
    bboxes_e1 = eyesDetector(face) ;
    bboxes_m = [bboxes_m; bboxes_m1+[x_s,y_s,0,0]] ;
    bboxes_e = [bboxes_e; bboxes_e1+[x_s,y_s,0,0]] ;
end

bboxes_final=[] ;
overlapbboxes =[] ;
for i = 1:size(bboxes_m,1)
    for j = 1:size(bboxes_e,1)
        overlapRatio = bboxOverlapRatio(bboxes_m(i,:),bboxes_e(j,:));
        if overlapRatio > 0.1
            overlapbboxes = [overlapbboxes,i] ;
        end
    end
end

for i=1:size(overlapbboxes,2)
    k = overlapbboxes(i)-i+1 ;
    bboxes_m(k,:) = [] ;
end

if isempty(bboxes_m) == 0
   % bboxes_m_1 = bboxes_m ;
    IMouth = insertObjectAnnotation(I,'rectangle',bboxes_m,'Mouth');
else
    % bboxes_m = bboxes_m_1 ;
    IMouth = I ;
end
