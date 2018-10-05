function [mean] = myMean(matrix)
%Computes the mean vector
r = size(matrix, 1);

mean = sum(matrix)/r;

end

