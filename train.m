%% Train our own Cascade detectors

% Get the positive samples(mouth)
positiveFolder='/True_false_data/positive_data/' ;

% Get the negative samples (eyes, nose, cheeks, chin)
negativeFolder='/Users/Pierre/Documents/Etats-Unis/CPE_428/Final Project/True_false_data/negative_data/' ;

% Preprocess the positives samples
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

% Use the prebuilt function to train the cascade classifier with HOG features
trainCascadeObjectDetector('lipsDetector1.xml',positiveInstances, negativeFolder,'FalseAlarmRate',0.3,'NumCascadeStages',2,'FeatureType','HOG');

myMouthDetector1 = vision.CascadeObjectDetector('lipsDetector1.xml') 

% Use the prebuilt function to train the cascade classifier with LBP features (To compare with our own LBP features)
trainCascadeObjectDetector('lipsDetector2.xml',positiveInstances, negativeFolder,'FalseAlarmRate',0.6,'NumCascadeStages',2,'FeatureType','LBP');


% Extract the LBP features 
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
 
myMouthDetector2 = fitcsvm(LBPTableFeatures,LabelTable);
