function timeComplexity=Classifier1(image)
% Load the classifier
faceDetector = vision.CascadeObjectDetector;
myMouthDetector1 = vision.CascadeObjectDetector('lipsDetector1.xml');
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
    bottomHalf = imColor(y_s:y_e,x_s:x_e);
    
    % Classifier 1: HOG Feature
    bboxes_m11 = myMouthDetector1(bottomHalf);
    bboxes_m21 = [bboxes_m21;bboxes_m11+[x_s,y_s,0,0]] ;
    
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
position = [bboxesBottomHalf; bboxes_m21];
num_face = size(bboxesBottomHalf, 1);
num_lips = size(bboxes_m21, 1);
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
figure, imshow(test) ;