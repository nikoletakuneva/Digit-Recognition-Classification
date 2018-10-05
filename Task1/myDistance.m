function [d] = myDistance(A, B)
%Computes the squared Euclidean distance between each row of matrix A and
%each row of matrix B
N = size(A, 1);
M = size(B, 1);
XX = sum(A.*A,2);
YY = sum(B.*B,2);
d = repmat(XX, 1, M) - 2 * A * B.' + (repmat(YY, 1, N)).';

end

