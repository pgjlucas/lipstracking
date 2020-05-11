function timeComplexity=Classifier3(image)
% Load the classifier
faceDetector = vision.CascadeObjectDetector;
myMouthDetector1 = vision.CascadeObjectDetector('lipsDetector1.xml');

positiveFolder='/Users/Pierre/Documents/Etats-Unis/CPE_428/Final Project/True_false_data/positive_data/' ;
negativeFolder='/Users/Pierre/Documents/Etats-Unis/CPE_428/Final Project/True_false_data/negative_data/' ;
negativeImages = imageDatastore(negativeFolder);
positiveImages = imageDatastore(positiveFolder);

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
% Image processing
imColor = image ;
imGray = rgb2gray(imColor) ;
bboxes = faceDetector(imColor);

% Time complexity measurement
tic;
initime = cputime;
time1   = clock;

% Classifier algorithm
bboxes_m11 = [] ;
bboxes_m21 = [] ;
bboxes_m31 = [] ;
bboxes_m41 = [] ;

% Select the bottom half
bboxesBottomHalf = [bboxes(:,1),bboxes(:,2)+(bboxes(:,4)/2),bboxes(:,3),(bboxes(:,4)/2)] ;

% Run the classifier algoithm on each faces
for i =1:size(bboxes,1)
    x_s = bboxes(i,1) ; y_s = bboxes(i,2) + (bboxes(i,4)/2) ;
    x_e = x_s + bboxes(i,3) ; y_e = y_s + (bboxes(i,4)/2) ;
    bottomHalf = imColor(y_s:y_e,x_s:x_e,:);
    
    % Classifier 1: HOG Feature
    bboxes_m11 = myMouthDetector1(bottomHalf);
    disp(size(bboxes_m11,1));

    if size(bboxes_m11, 1) > 1
        for j = 1:size(bboxes_m11, 1) 
            imgLip = imColor(bboxes_m11(j,2)+y_s:bboxes_m11(j,2)+y_s+bboxes_m11(j,4), ...
                bboxes_m11(j,1)+x_s:x_s+bboxes_m11(j,1)+bboxes_m11(j,3), :);
            imgLipResize = imresize(imgLip,[68 130]);

            % Classifier 3: LBP Feature
            imgLipLBP = extractLBPFeatures(rgb2gray(imgLipResize)) ;
            if predict(MyMouthDetector2,imgLipLBP) == 1
                bboxes_m41 = [bboxes_m41;bboxes_m11(j,:)+[x_s,y_s,0,0]];
            end
        end
    else
        bboxes_m41 = [bboxes_m41;bboxes_m11+[x_s,y_s,0,0]] ; 
    end  
end
      
% Time complexity measurement
fintime = cputime;
elapsed = toc;
time2   = clock;
fprintf('TIC TOC: %g\n', elapsed);
fprintf('CPUTIME: %g\n', fintime - initime);
fprintf('CLOCK:   %g\n', etime(time2, time1))
timeComplexity = [elapsed, fintime - initime, etime(time2, time1)];
% Display the results
position = [bboxesBottomHalf; bboxes_m41];
num_face = size(bboxesBottomHalf, 1);
num_lips = size(bboxes_m41, 1);
label = {};
color = {};
for i = 1:num_face
    label{end+1} = 'Bottom face';
    color{end+1} = 'yellow';
end

for i = 1:num_lips
    label{end+1} = 'lips';
    color{end+1} = 'cyan';
end
test = insertObjectAnnotation(imColor,'rectangle',position,label,'LineWidth',1,'Color',color,'TextColor','black');
figure, imshow(test)
