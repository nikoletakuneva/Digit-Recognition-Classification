%
% Template for my_knn_system.m
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

%YourCode - Prepare measuring time
tic

% Run K-NN classification
kb = [1,3,5,10,20];
Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, kb.');

%YourCode - Measure the time taken, and display it.
toc

%YourCode - Get confusion matrix and accuracy for each k in kb.
for i=1: length(kb)
    k = kb(i);
    [cm,acc] = my_confusion(Ctst, Cpreds(:, i));
    
    %YourCode - Save each confusion matrix.
    fname = sprintf ( '%s%i', 'cm', k );
    save(fname, 'cm');

    %YourCode - Display the required information - k, N, Nerrs, acc for
    %           each element of kb.
    display = sprintf('K: %d,  N: %d,  Number of errors: %d,  Accuracy: %.4f', k, size(Xtst,1), sum(sum(cm)) - trace(cm), acc);
    disp(display);
    
end




  
