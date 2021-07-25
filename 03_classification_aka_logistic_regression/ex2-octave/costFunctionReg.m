function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
h = @sigmoid;
observations_q = rows(y);
m = observations_q;
features_q = columns(X);

%applyFnToRow = @(func, matrix) @(row) func(matrix(row,:))
%applyFnToAllRows = @(func, matrix) arrayfun(applyFnToRow(func, matrix), 1:rows(matrix));
%applyFnToCol = @(func, matrix) @(col) func(matrix(:,col))
%applyFnToAllCols = @(func, matrix) arrayfun(applyFnToCol(func, matrix), 1:columns(matrix));


% gen h(x) vector
x = X*theta;
h_x = h(x);

%mask for theta (ie. leave out theta 0)
theta_mask = ones(rows(theta),1);
theta_mask(1) = 0;
theta_masked = theta .* theta_mask;


% cost function
sum_cost = 0;
for i = 1:observations_q
    sum_cost += -y(i)*log(h_x(i))-(1-y(i))*log(1-h_x(i));
endfor
J = (1/m)*sum_cost;
J += (lambda/(2*m))*(sum(theta_masked.^2));



% gradient
sum_grad = 0;
error_v = h_x .- y;
for i = 1:observations_q
        error_i = error_v(i);
        sum_grad += (error_i.*X(i,:))';
endfor
sum_grad += (theta_masked .* lambda);


grad = ((1/m).*sum_grad);

% don't know why it should be negative, but ok
%grad = grad .* -1;

% =============================================================
end
