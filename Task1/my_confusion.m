function [CM, acc] = my_confusion(Ctrues, Cpreds)
% Input:
%   Ctrues : N-by-1 ground truth label vector
%   Cpreds : N-by-1 predicted label vector
% Output:
%   CM : K-by-K confusion matrix, where CM(i,j) is the number of samples whose target is the ith class that was classified as j
%   acc : accuracy (i.e. correct classification rate)

K = length(unique(Ctrues));
N = size(Ctrues, 1);

CM = zeros(K,K);
for i=1: size(Ctrues,1)
   CM(Ctrues(i,1), Cpreds(i,1)) = CM(Ctrues(i,1), Cpreds(i,1)) + 1;
end

acc = trace(CM)/N;

end
