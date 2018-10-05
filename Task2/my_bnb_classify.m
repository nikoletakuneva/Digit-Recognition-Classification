function [Cpreds] = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold)
% Input:
%   Xtrn : M-by-D training data matrix
%   Ctrn : M-by-1 label vector for Xtrn
%   Xtst : N-by-D test data matrix
%   threshold : A scalar parameter for binarisation
% Output:
%  Cpreds : N-by-1 matrix of predicted labels for Xtst

%YourCode - binarisation of Xtrn and Xtst.
M = size(Xtrn, 1);
N = size(Xtst, 1);
D = size(Xtrn, 2);
K = length(unique(Ctrn));

binary_Xtrn = Xtrn >= threshold;
binary_Xtst = Xtst >= threshold;

%YourCode - naive Bayes classification with multivariate Bernoulli distributions
M_class = M/K;
doc_count = zeros(D, K);

% Computes the number of samples of a given class containing a given
% feature.
for class=1:K
    for feature=1:D
        for s_num=1:M
            if Ctrn(s_num) == class
                doc_count(feature, class) = doc_count(feature, class) + binary_Xtrn(s_num, feature);
            end
        end
    end
end

likelihood_matrix = doc_count ./ M_class;

post_probability = zeros(K, N);

% Computes the posterior probability for each class for each sample of the
% test dataset.
for s_num=1:N
    for class=1:K
        product = 1;
        for feature=1:D
            b = binary_Xtst(s_num, feature);
            P = likelihood_matrix(feature, class);
            x = (b * P + (1-b) * (1-P));
            
            if x == 0
                x = 1.0E-10;
            end
            
            product = product * x;
        end
        post_probability(class, s_num) = product;
    end
end
    
[~, Cpreds] = max(post_probability.', [], 2);

end
