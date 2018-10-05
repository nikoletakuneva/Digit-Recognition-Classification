%
% Template for my_gaussian_system.m
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
epsilon = 0.01;
[Cpreds, Ms, Covs] = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon);

%YourCode - Measure the time taken, and display it.
toc

%YourCode - Get a confusion matrix and accuracy
[cm, acc] = my_confusion(Ctst, Cpreds);

%YourCode - Save the confusion matrix as "Task3/cm.mat".
save('cm.mat', 'cm');

%YourCode - Save the mean vector and covariance matrix for class 26,
%           i.e. save Mu(:,26) and Cov(:,:,26) as "Task3/m26.mat" and
%           "Task3/cov26.mat", respectively.
m26 = Ms(:,26);
cov26 = Covs(:,:,26);
save('m26.mat', 'm26');
save('cov26.mat', 'cov26');

%YourCode - Display the required information - N, Nerrs, acc.

display = sprintf('N: %d,  Number of errors: %d,  Accuracy: %.4f', size(Xtst,1), sum(sum(cm)) - trace(cm), acc);
disp(display);




  
