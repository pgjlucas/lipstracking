totalTime = [0,0,0] ;
for i=6:6
    filename =['/Users/Pierre/Documents/Etats-Unis/CPE_428/Final Project/Test/img' num2str(i) '.jpg'];
    timeComplexity=Classifier3(imread(filename));
    totalTime = totalTime + timeComplexity;
end

averageTime = totalTime/25 ;