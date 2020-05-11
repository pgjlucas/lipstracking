% Compute time complexity for the classifier3
totalTime = [0,0,0] ;
for i=6:6
    filename =['/path/img/' num2str(i) '.jpg'];
    timeComplexity=Classifier3(imread(filename));
    totalTime = totalTime + timeComplexity;
end

averageTime = totalTime/25 ;
