function [Cpreds, Ms, Covs] = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon)
% Input:
%   Xtrn : M-by-D training data matrix
%   Ctrn : M-by-1 label vector for Xtrn
%   Xtst : N-by-D test data matrix
%   epsilon : A scalar parameter for regularisation
% Output:
%  Cpreds : N-by-1 matrix of predicted labels for Xtst
%  Ms    : D-by-K matrix of mean vectors
%  Covs  : D-by-D-by-K 3D array of covariance matrices

%YourCode - Bayes classification with multivariate Gaussian distributions.
M = size(Xtrn, 1);
N = size(Xtst, 1);
D = size(Xtrn, 2);
K = length(unique(Ctrn));

Ms = zeros(D,K);
Covs = zeros(D, D, K);

% Writes the means and covariance matrices for each class in Ms and Covs
% respectively.
for class=1:K 
  train_class = Xtrn((Ctrn == class), :); % matrix containing all the train samples of a given class
  mu = myMean(train_class);
  sigma = myCov(train_class, mu);
  Ms(:, class) = mu';
  Covs(:, :, class) = sigma;
end

post_probability = zeros(K, N);

% Computes the posterior probability for each class for each sample of the
% test dataset.
for class=1:K
    mu = Ms(:, class);
    mu = mu.';
    sigma = Covs(:, :, class);
    sigma = sigma + epsilon * eye(D);
    X = Xtst - ones(N, 1) * mu;
    fact = sum(((X * inv(sigma)) .* X), 2);
    post_probability(class, :) = -0.5 * fact - 0.5 * logdet(sigma) + log(1/K);
end

[~, Cpreds] = max(post_probability.', [], 2);

end
