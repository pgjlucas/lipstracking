% Baseline Face: first, detect the faces
faceDetector = vision.CascadeObjectDetector;
imGray = rgb2gray(c) ;
im = imGray(1:1:end,1:1:end);
bboxes = myMouthDetector2(im);
IFaces = insertObjectAnnotation(im,'rectangle',bboxes,'Face');   
figure
imshow(IFaces)
title('Detected faces');

% Baseline Mouth: then, detect the mouth from the faces
bboxes_m11 =[] ;
bboxes_m12 = [] ;
for i =1:size(bboxes,1)
    x_s = bboxes(i,1) ; y_s = bboxes(i,2) ; x_e = x_s + bboxes(i,3) ; y_e = y_s + bboxes(i,4) ;
    bottomHalf = im(y_s:y_e,x_s:x_e);
    mouthDetector = vision.CascadeObjectDetector('Mouth','MergeThreshold',20) ;
    bboxes_m11 = mouthDetector(bottomHalf);
    bboxes_m12 = [bboxes_m12; bboxes_m11+[x_s,y_s,0,0]] ;

end
IMouth = insertObjectAnnotation(im,'rectangle',bboxes_m12,'Mouth');   
figure
imshow(IMouth)
title('Detected mouth');


% We have many false positive, eyes being labelled as mouth. In order to
% correct that, we decided to detect eyes and apply an IoU. 

bboxes_m11 =[] ;
bboxes_e1 = [] ;
bboxes_m12 = [] ;
bboxes_e = [] ;
for i = 1:size(bboxes,1)
    x_s = bboxes(i,1) ; y_s = bboxes(i,2) ; x_e = x_s + bboxes(i,3) ; y_e = y_s + bboxes(i,4) ;
    bottomHalf = im(y_s:y_e,x_s:x_e);
    mouthDetector = vision.CascadeObjectDetector('Mouth','MergeThreshold',20) ;
    eyesDetector = vision.CascadeObjectDetector('EyePairBig') ;
    bboxes_m11 = mouthDetector(bottomHalf);
    bboxes_e1 = eyesDetector(bottomHalf) ;
    bboxes_m12 = [bboxes_m12; bboxes_m11+[x_s,y_s,0,0]] ;
    bboxes_e = [bboxes_e; bboxes_e1+[x_s,y_s,0,0]] ;
end

bboxes_final=[] ;
overlapbboxes =[] ;
for i = 1:size(bboxes_m12,1)
    for j = 1:size(bboxes_e,1)
        overlapRatio = bboxOverlapRatio(bboxes_m12(i,:),bboxes_e(j,:));
        if overlapRatio > 0.1
            overlapbboxes = [overlapbboxes,i] ;
        end
    end
end

for i=1:size(overlapbboxes,2)
    k = overlapbboxes(i)-i+1 ;
    bboxes_m12(k,:) = [] ;
end
IMouth = insertObjectAnnotation(im,'rectangle',bboxes_m12,'Mouth');   
figure
imshow(IMouth)
title('Detected mouth');

% Then, in order to track the mouthes, we implemented an algorithm that 
% kept in memory the previous position of the mouth, in order to reduce 
% the time complexity.

%% VIDEO READER
%video = read(v,1);

v = VideoReader('/Users/Pierre/Documents/Etats-Unis/CPE_428/Final Project/id2_vcd_swwp2s.mpg'); 

vid = VideoWriter('video_color.avi') ;
open(vid) ;
while hasFrame(v)
    frame = readFrame(v);
    frame = my_detectedMouth(frame,1,16);
    writeVideo(vid,frame);
end
close(vid);



%% Train our own Cascade detector

positiveFolder='/Users/Pierre/Documents/Etats-Unis/CPE_428/Final Project/True_false_data/positive_data/' ;

negativeFolder='/Users/Pierre/Documents/Etats-Unis/CPE_428/Final Project/True_false_data/negative_data/' ;

positiveInstances = table ;
boxSizes ={} ;
fileNamesPos = {} ;
for i = 1:1:160
    fileNamesPos(i)= {[positiveFolder 'lip' num2str(i) '.png']};
    boxSizes(i)= {[1,1,130,68]};
end
positiveInstances=table(fileNamesPos',boxSizes');

negativeImages = imageDatastore(negativeFolder);
positiveImages = imageDatastore(positiveFolder);

trainCascadeObjectDetector('lipsDetector1.xml',positiveInstances, negativeFolder,'FalseAlarmRate',0.3,'NumCascadeStages',2,'FeatureType','HOG');
trainCascadeObjectDetector('lipsDetector2.xml',positiveInstances, negativeFolder,'FalseAlarmRate',0.6,'NumCascadeStages',2,'FeatureType','LBP');

myMouthDetector1 = vision.CascadeObjectDetector('lipsDetector1.xml');
myMouthDetector2 = vision.CascadeObjectDetector('lipsDetector2.xml');

LBPTableFeatures = [] ;
for i=1:160
    LBPTableFeatures(i,:) = extractLBPFeatures(rgb2gray(readimage(positiveImages,i)));
    LBPTableFeatures(i+160,:) = extractLBPFeatures(rgb2gray(readimage(negativeImages,i)));
end

LabelTable = [] ;
for i=1:160
    LabelTable(i,1) = 1;
    LabelTable(i+160,1) = 0;
end

MyMouthDetector2 = fitcsvm(LBPTableFeatures,LabelTable);
l = extractHOGFeatures(cdata);
u = predict(a,l) ;


%% Try it on the bottom half of the face

% Baseline Face: first, detect the faces
faceDetector = vision.CascadeObjectDetector;
 
%% HOG + LBP

% Image processing
img = JM5_35U7;
imGray = rgb2gray(img);
im = imGray(1:1:end,1:1:end);
bboxes = faceDetector(im);

% Time measurement
tic;
initime = cputime;
time1   = clock;

% Classifier

bboxes_m11 =[] ;
bboxes_m12 = [] ;
bboxes_m21 = [] ;
bboxesBottomHalf = [bboxes(:,1),bboxes(:,2)+(bboxes(:,4)/2),bboxes(:,3),(bboxes(:,4)/2)] ;
for i =1:size(bboxes,1)
    x_s = bboxes(i,1) ; y_s = bboxes(i,2) + (bboxes(i,4)/2) ;
    x_e = x_s + bboxes(i,3) ; y_e = y_s + (bboxes(i,4)/2) ;
    bottomHalf = im(y_s:y_e,x_s:x_e);
    bboxes_m11 = myMouthDetector1(bottomHalf); 
    bboxes_m21 = myMouthDetector2(bottomHalf);
    
    if size(bboxes_m11,1) == 1
        bboxes_m12 = [bboxes_m12; bboxes_m11+[x_s,y_s,0,0]] ;
    end
    
    if size(bboxes_m11,1) < 1 && size(bboxes_m21,1) == 1
        bboxes_m12 = [bboxes_m12; bboxes_m21+[x_s,y_s,0,0]] ;    
    end
    
    if size(bboxes_m11,1) < 1 && size(bboxes_m21,1) > 1 
        a = 0 ;
        for j=1:size(bboxes_m21,1)
            for k=1:size(bboxes_m11,1)
                l = j- a ;
                if bboxOverlapRatio(bboxes_m21(l,:),bboxes_m11(k,:)) < 0.4
                    bboxes_m21(l,:) = [] ;
                    a = a + 1 ;
                end
                if size(bboxes_m21,1) == 1
                    break
                end
            end
        end
        bboxes_m12 = [bboxes_m12; bboxes_m21+[x_s,y_s,0,0]] ;
    end
    
    if size(bboxes_m11,1) > 1
        a = 0 ;
        for j=1:size(bboxes_m11,1)
            for k=1:size(bboxes_m21,1)
                l = j- a ;
                if bboxOverlapRatio(bboxes_m11(l,:),bboxes_m21(k,:)) < 0.4
                    bboxes_m11(l,:) = [] ;
                    a = a + 1 ;
                end
                if size(bboxes_m11,1) == 1
                    break
                end
            end
        end
        bboxes_m12 = [bboxes_m12; bboxes_m11+[x_s,y_s,0,0]] ;
    end     
end

fintime = cputime;
elapsed = toc;
time2   = clock;
fprintf('TIC TOC: %g\n', elapsed);
fprintf('CPUTIME: %g\n', fintime - initime);
fprintf('CLOCK:   %g\n', etime(time2, time1));

figure
IFaces = insertObjectAnnotation(im,'rectangle',bboxesBottomHalf,'Bottom Face'); 
imshow(IFaces)
hold on 
IMouth = insertObjectAnnotation(img,'rectangle',bboxes_m12,''); %IFaces instead of img
imshow(IMouth)
title('Detected mouth');

%%

%% HOG + LBP

img = IM5_35D1 ;
imGray = rgb2gray(img) ;
im = imGray(1:1:end,1:1:end);
bboxes = faceDetector(im);

tic;
initime = cputime;
time1   = clock;
bboxes_m11 =[] ;
bboxes_m12 = [] ;
bboxesBottomHalf = [bboxes(:,1),bboxes(:,2)+(bboxes(:,4)/2),bboxes(:,3),(bboxes(:,4)/2)] ;
for i =1:size(bboxes,1)
    x_s = bboxes(i,1) ; y_s = bboxes(i,2) + (bboxes(i,4)/2) ;
    x_e = x_s + bboxes(i,3) ; y_e = y_s + (bboxes(i,4)/2) ;
    bottomHalf = im(y_s:y_e,x_s:x_e);
    bboxes_m11 = myMouthDetector1(bottomHalf); 
    bboxes_m12 = [bboxes_m12; bboxes_m11+[x_s,y_s,0,0]] ;  
end

fintime = cputime;
elapsed = toc;
time2   = clock;
fprintf('TIC TOC: %g\n', elapsed);
fprintf('CPUTIME: %g\n', fintime - initime);
fprintf('CLOCK:   %g\n', etime(time2, time1));

figure
%IFaces = insertObjectAnnotation(im,'rectangle',bboxesBottomHalf,'Bottom Face'); 
%imshow(IFaces)
%hold on 
IMouth = insertObjectAnnotation(img,'rectangle',bboxes_m12,''); 
imshow(IMouth)
title('Detected mouth');

%%

% Image processing
img = JM5_35U7 ;
imGray = rgb2gray(img) ;
im = imGray(1:1:end,1:1:end);
bboxes = faceDetector(im);

% Time complexity measurement
tic;
initime = cputime;
time1   = clock;

% Classifier algorithm
bboxes_m11 =[] ;
bboxes_m21 = [] ;
bboxes_m31 = [] ;
% Select the bottom half
bboxesBottomHalf = [bboxes(:,1),bboxes(:,2)+(bboxes(:,4)/2),bboxes(:,3),(bboxes(:,4)/2)] ;

% Run the classifier algoithm on each faces
for i =1:size(bboxesBottomHalf,1)
    x_s = bboxes(i,1) ; y_s = bboxes(i,2) + (bboxes(i,4)/2) ;
    x_e = x_s + bboxes(i,3) ; y_e = y_s + (bboxes(i,4)/2) ;
    bottomHalf = im(y_s:y_e,x_s:x_e);
    
    % Classifier 1: HOG Feature
    bboxes_m11 = myMouthDetector1(bottomHalf); 
    
    if size(bboxes_m11, 1) > 1
        for n = 1:size(bboxes_m11, 1)
         
            imgLip = bottomHalf(bboxes_m11(n,2)+y_s:bboxes_m11(n,2)+y_s+bboxes_m11(n,4), ...
                bboxes_m11(n,1)+x_s:bboxes_m11(n,1)+x_s+bboxes_m11(n,3), :);
            
            % Classifier 2: Color Feature
            YCbCr = rgb2ycbcr(double(imgLip)/256);
            Y  = double(YCbCr(:, :, 1));
            Cb = double(YCbCr(:, :, 2));
            Cr = double(YCbCr(:, :, 3));
            
            HSI = rgb2hsv(imgLip);
            S = HSI(:, :, 2);
            
            c = 0.95 * sum(sum(Cr.^2)) / sum(sum(Cr./Cb));
            lip_map = (S.^0) .* (Cr.^2) .* ( Cr.^2 - c * Cr./Cb ).^2;

            judge(n) = std(lip_map(:));
            
            % Classifier 3: LBP Feature
            bboxes_m31 = myMouthDetector2(imgLip);
        end
        [~, index] = max(judge);
        bboxes_m1 = bboxes_m1(index, :);
    
        for j=1:size(bboxes_m21,1)
            %for k=1:size(bboxes_m11,1)
            l = j- a ;
            if bboxOverlapRatio(bboxes_m21(l,:),bboxes_m11(k,:)) < 0.4
                bboxes_m21(l,:) = [] ;
                a = a + 1 ;
            end
                if size(bboxes_m21,1) == 1
                    break
                end
            end
        end
        bboxes_m12 = [bboxes_m12; bboxes_m21+[x_s,y_s,0,0]] ;
    end
    
            

            
            
       
        [~, index] = max(judge);
        bboxes_m1 = bboxes_m1(index, :);
    end
        
    bboxes_m12 = [bboxes_m12; bboxes_m11+[x_s,y_s,0,0]] ;
    
end

fintime = cputime;
elapsed = toc;
time2   = clock;
fprintf('TIC TOC: %g\n', elapsed);
fprintf('CPUTIME: %g\n', fintime - initime);
fprintf('CLOCK:   %g\n', etime(time2, time1))

figure
IFaces = insertObjectAnnotation(im,'rectangle',bboxesBottomHalf,'Bottom Face'); 
imshow(IFaces)
hold on 
IMouth = insertObjectAnnotation(IFaces,'rectangle',bboxes_m12,''); 
imshow(IMouth)
title('Detected mouth');
%% DISPLAY HOG FEATURES
% Select a picture from DataSet
imGray = rgb2gray(JM5_35U7) ;
im = imGray(1:1:end,1:1:end);
% Apply FaceDetector
bboxes = faceDetector(im);
% Choose 1 face
i = 2 ;
x_s = bboxes(i,1) ; y_s = bboxes(i,2) + (bboxes(i,4)/2) ;
x_e = x_s + bboxes(i,3) ; y_e = y_s + (bboxes(i,4)/2) ;
face = im(y_s:y_e,x_s:x_e) ;
[featureVector,hogVisualization] = extractHOGFeatures(face,'CellSize',[6 6]);
figure;
imshow(face); 
hold on;
plot(hogVisualization);

%% Extract HOG Features from Data

HOGFeatureTable = table ;

for i = 1:1:160
    imgPos = readimage(imds,I)
    imgNeg = readimage(imds,I)
    HOGPos = extractHOGFeatures(imgPos)
    HOGNeg = extractHOGFeatures(imgNeg)
    HOGFeatureTable = [
end

%% DISPLAY LBP FEATURES

% Select a picture from DataSet
imGray = rgb2gray(JM5_35U7) ;
im = imGray(1:1:end,1:1:end);
% Apply FaceDetector
bboxes = faceDetector(im);
% Choose 1 face
i = 2 ;
x_s = bboxes(i,1) ; y_s = bboxes(i,2) + (bboxes(i,4)/2) ;
x_e = x_s + bboxes(i,3) ; y_e = y_s + (bboxes(i,4)/2) ;
face = im(y_s:y_e,x_s:x_e) ;
featureVectorLBP = extractLBPFeatures(face) ;
B = reshape(featureVectorLBP,size(face,1),size(face,2)) ;
figure;
imshow(B); 


%%

% if DoUniform = true -> return hisogram of 10 bin,  if DoUniform = false -> return hisogram of 256 bin


OrgIm=face;
Row=size(OrgIm,1);
Col=size(OrgIm,2);
DoUniform = 0 ;
for i=2:Row-1
    for j=2:Col-1
        Uniform = true;
        MidPixelValue=OrgIm(i,j);
        EncodedVec(1)=OrgIm(i-1,j-1)>MidPixelValue;
        EncodedVec(2)=OrgIm(i-1,j)>MidPixelValue;
        EncodedVec(3)=OrgIm(i-1,j+1)>MidPixelValue;
        EncodedVec(4)=OrgIm(i,j+1)>MidPixelValue;
        EncodedVec(5)=OrgIm(i+1,j+1)>MidPixelValue;
        EncodedVec(6)=OrgIm(i+1,j)>MidPixelValue;
        EncodedVec(7)=OrgIm(i+1,j-1)>MidPixelValue;
        EncodedVec(8)=OrgIm(i,j-1)>MidPixelValue;
        EncodedVecShift = circshift(EncodedVec,[0,1]);
        if DoUniform
            if sum(xor(EncodedVec,EncodedVecShift)) > 2 % more than 2 transition of 0 -> 1
                Uniform = false;
                LBPImage(i,j)=9;
            end
        end
        if or(Uniform == true  , DoUniform == false) % if LBP not uniform mode , or the texture is uniform -> 8 bits assign
            MinLbp = EncodedVec(1)*2^7+EncodedVec(2)*2^6+EncodedVec(3)*2^5+EncodedVec(4)*2^4+EncodedVec(5)*2^3+EncodedVec(6)*2^2+EncodedVec(7)*2^1+EncodedVec(8)*2^0;
            MinVector = EncodedVec;
            for k = 1 : 7
                EncodedVec = circshift(EncodedVec,[0,1]);
                CurrLbpValue =EncodedVec(1)*2^7+EncodedVec(2)*2^6+EncodedVec(3)*2^5+EncodedVec(4)*2^4+EncodedVec(5)*2^3+EncodedVec(6)*2^2+EncodedVec(7)*2^1+EncodedVec(8)*2^0;
                if CurrLbpValue < MinLbp
                    MinLbp = CurrLbpValue;
                    MinVector = EncodedVec;
                end
            end
            LBPImage(i,j)=MinVector(1)*2^7+MinVector(2)*2^6+MinVector(3)*2^5+MinVector(4)*2^4+MinVector(5)*2^3+MinVector(6)*2^2+MinVector(7)*2^1+MinVector(8)*2^0;
        end
    end
end

imshow(LBPImage) ;
%% 
A = uint8(LBPImage) ;
B = A + face(1:76,1:153); 
imshow(B);
face+I);
for i = 1:a=face+I ;
