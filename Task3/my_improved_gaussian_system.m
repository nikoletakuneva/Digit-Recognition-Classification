%
% Template for my_improved_gaussian_system.m
%
% load the data set
%   NB: replace <UUN> with your actual UUN.
load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1643102/data.mat');

% Feature vectors: Convert uint8 data to double, and divide by 255.
Xtrn = double(dataset.train.images) ./ 255.0;
Xtst = double(dataset.test.images) ./ 255.0;
% Labels
Ctrn = dataset.train.labels;
Ctst = dataset.test.labels;

%YourCode - Prepare to measure time
tic

% Run classification
epsilon = 0.03;
k = 2;
[Cpreds] = my_improved_gaussian_classify(Xtrn, Ctrn, Xtst, k, epsilon);

%YourCode - Measure the time taken, and display it.
toc

%YourCode - Get a confusion matrix and accuracy
[cm, acc] = my_confusion(Ctst, Cpreds);

%YourCode - Save the confusion matrix as "Task3/cm_improved.mat".
save('cm_improved.mat', 'cm');

%YourCode - Display information if any

display = sprintf('Epsilon: %f,  Number of clusters: %d,   N: %d,  Number of errors: %d,  Accuracy: %.4f', epsilon, k, size(Xtst,1), sum(sum(cm)) - trace(cm), acc);
disp(display);




  
