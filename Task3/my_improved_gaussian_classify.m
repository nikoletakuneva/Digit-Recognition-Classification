function [Cpreds] = my_improved_gaussian_classify(Xtrn, Ctrn, Xtst, k, epsilon)
% Input:
%   Xtrn : M-by-D training data matrix
%   Ctrn : M-by-1 label vector for Xtrn
%   Xtst : N-by-D test data matrix
%   NB: you may add arguments if necessary
%   k : number of clusters
%   epsilon : A scalar parameter for regularisation

% Output:
%  Cpreds : N-by-1 matrix of predicted labels for Xtst

%YourCode
M = size(Xtrn, 1);
N = size(Xtst, 1);
D = size(Xtrn, 2);
K = length(unique(Ctrn));
preds = zeros(1, N);

c_centres = zeros(k, D);
for cluster=1:k
    % Choose the first k data train samples as centres
    c_centres(cluster, :) = Xtrn(cluster, :);
end

maxiter = 50; %maximum number of iterations
for i = 1:maxiter
    % Compute Squared Euclidean distance (i.e. the squared distance)
    % between each cluster centre and each observation
    DI = myDistance(Xtrn, c_centres);
    [~, idx] = min(DI,[],2);
    % Update cluster centres
    for cluster = 1:k
        c_centres(cluster, :) = mean(Xtrn(idx==cluster, :));
    end
end

test_cluster_num = zeros(N, 1);

for s_num=1:N
    %Stores the cluster of each test sample based on the distance to the cluster centres
    sample = Xtst(s_num, :);
    d = myDistance(c_centres, sample);
    [~, i] = min(d);
    test_cluster_num(s_num, 1) = i;
end

train_cluster_num = zeros(M, 1);

for s_num=1:M
    %Stores the cluster of each train sample based on the distance to the cluster centres
    sample = Xtrn(s_num, :);
    d = myDistance(c_centres, sample);
    [~, i] = min(d);
    train_cluster_num(s_num, 1) = i;
end

for cluster=1:k
    train_indices = find(train_cluster_num == cluster); % indices of the train samples belonging to the given cluster
    
    test_indices = find(test_cluster_num == cluster); % indices of the test samples belonging to the given cluster
    test_cluster = Xtst(test_indices, :); % a matrix containing the test samples belonging to the given cluster
    cluster_size = size(test_cluster, 1);
    post_probability = zeros(K, cluster_size);
    
    % Computes the posterior probability for each class for each sample of the test dataset belonging to the given cluster.
    for class=1:K
        class_indices = find(Ctrn == class);
        train_class = Xtrn(intersect(train_indices, class_indices), :); % a matrix containing all the test samples belonging to the given cluster and the given class
        if isempty(train_class)
            post_probability(class, :) = 0;
            continue
        end
        if size(train_class,1)==1
            post_probability(class, :) = 1.0;
            continue
        end
        
        mu = myMean(train_class);
        sigma = myCov(train_class, mu);
        sigma = sigma + epsilon * eye(D);
        
        X = test_cluster - ones(cluster_size, 1)*mu;
        fact = sum(((X * inv(sigma)) .* X), 2);
        post_probability(class, :) = -0.5 * fact - 0.5 * logdet(sigma) + log(1/K);
    end
    
    % Add the predictions for the samples belonging to the given cluster
    [~, preds(test_indices)] = max(post_probability.', [], 2);
    
end

Cpreds = preds.';

end
