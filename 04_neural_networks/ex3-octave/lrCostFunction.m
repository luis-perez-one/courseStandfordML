function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with  regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the gradient of the cost w.r.t. to the parameters. 
    %% theta : matrix of weights controlling function mapping from layer j to layer j+1

g=@sigmoid;
m = rows(y);
%h_x: Each row of the resulting matrix will contain the value of the
h_x = g(X*theta);
J=(1/m)*(sum((-y.*log(h_x))-((1.-y).*log(1-(h_x)))));
grad = (1/m)*X'*(h_x-y);


% regularization
    % theta 0 (bias) should not be regularized, hence the mask
theta_mask = ones(rows(theta),1);
theta_mask(1) = 0;
theta_masked = theta .* theta_mask;

J += (lambda/(2*m))*(sum(theta_masked.^2));
grad += (theta_masked .* (lambda/m));

end

