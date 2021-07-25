function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values


% You need to return the following variables correctly 
%J = 0;
%grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

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

% cost function
sum_cost = 0;
for i = 1:observations_q
    sum_cost += -y(i)*log(h_x(i))-(1-y(i))*log(1-h_x(i));
endfor
J = (1/m)*sum_cost;

% gradient
sum_grad = 0;
error_v = h_x .- y;
for i = 1:observations_q
        error_i = error_v(i);
        sum_grad += error_i.*X(i,:);
endfor
grad = ((1/m).*sum_grad)';

% =============================================================
end
