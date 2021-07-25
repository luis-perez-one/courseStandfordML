function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 

lrCost = @lrCostFunction;
% all_theta must be returned
% initializtaion of all_theta
%all_theta = zeros(num_labels, n + 1);
% Add bias neurons to X
obs_q = m = rows(X);
feat_q = n = columns(X);
X = [ones(m, 1) X];
all_theta = [];

% iterate over all the classes and estimate new theta parameters for each class using fmincg to minimize the cost.
% the new theta parameters are the gradient
% each class theta parameters should be appended to the all_theta matrix
% options for fmincg
max_iter = 400;
options = optimset('GradObj', 'on', 'MaxIter', max_iter);

for k = 1:num_labels
    initial_theta = rand(feat_q+1,1);
    initial_theta = initial_theta.-0.5; %negatives and positives.
    initial_theta = 1./initial_theta;
    [learned_theta] = fmincg(@(t)(lrCost(t, X, (y==k), lambda)), initial_theta(:,1), options);
    all_theta = [all_theta learned_theta];
end
all_theta = all_theta';
csvwrite('all_theta.txt', all_theta);

%size(all_theta)
end




% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%












% =========================================================================



