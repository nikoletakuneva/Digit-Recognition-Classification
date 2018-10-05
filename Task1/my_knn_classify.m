function [Cpreds] = my_knn_classify(Xtrn, Ctrn, Xtst, Ks)
% Input:
%   Xtrn : M-by-D training data matrix
%   Ctrn : M-by-1 label vector for Xtrn
%   Xtst : N-by-D test data matrix
%   Ks   : L-by-1 vector of the numbers of nearest neighbours in Xtrn
% Output:
%  Cpreds : N-by-L matrix of predicted labels for Xtst

N = size(Xtst, 1);
L = size(Ks, 1);
Cpreds = zeros(N, L);

DI = myDistance(Xtst, Xtrn);

[~, idx] = sort(DI, 2, 'ascend');

for i=1:L
    k = Ks(i, 1);
    idx_neighbours =  idx(:, 1:k); % get indices of the k nearest neighbours
    prediction = mode(Ctrn(idx_neighbours), 2);
    Cpreds(:, i) = prediction;
end
   
end
