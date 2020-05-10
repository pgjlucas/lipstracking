function IMouth=my_detectedMouthBaseLine(img,downSamplingRatio, thresholdMouth)
imGray = rgb2gray(img) ;
im = imGray(1:downSamplingRatio:end,1:downSamplingRatio:end);
faceDetector = vision.CascadeObjectDetector;
bboxes = faceDetector(im);
myMouthDetector1 = vision.CascadeObjectDetector('lipsDetector1.xml');


positiveFolder='/Users/Pierre/Documents/Etats-Unis/CPE_428/Final Project/True_false_data/positive_data/' ;

negativeFolder='/Users/Pierre/Documents/Etats-Unis/CPE_428/Final Project/True_false_data/negative_data' ;

positiveInstances = table ;
boxSizes ={} ;
fileNames = {} ;
for i = 1:1:80
    fileNames(i)= {[positiveFolder 'lip' num2str(i) '.png']};
    boxSizes(i)= {[1,1,130,68]};
end
positiveInstances=table(fileNames',boxSizes');

negativeImages = imageDatastore(negativeFolder);

trainCascadeObjectDetector('lipsDetector1.xml',positiveInstances, negativeFolder,'FalseAlarmRate',0.5,'NumCascadeStages',6,'FeatureType','HOG');
trainCascadeObjectDetector('lipsDetector2.xml',positiveInstances, negativeFolder,'FalseAlarmRate',0.5,'NumCascadeStages',4,'FeatureType','LBP');

myMouthDetector1 = vision.CascadeObjectDetector('lipsDetector1.xml');
myMouthDetector2 = vision.CascadeObjectDetector('lipsDetector2.xml');


bboxes_m11 =[] ;
bboxes_m12 = [] ;
bboxes_m21 = [] ;
bboxesBottomHalf = [bboxes(:,1),bboxes(:,2)+(bboxes(:,4)/2),bboxes(:,3),(bboxes(:,4)/2)] ;
for i =1:size(bboxes,1)
    if isempty(bboxes) == 0
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
                    if bboxOverlapRatio(bboxes_m21(l,:),bboxes_m11(k,:)) < thresholdMouth
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
                    if bboxOverlapRatio(bboxes_m11(l,:),bboxes_m21(k,:)) < thresholdMouth
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
end