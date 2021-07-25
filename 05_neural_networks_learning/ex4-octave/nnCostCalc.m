function [J] = nnCostCalc(theta1, theta2, num_labels, X, y, lambda)
% calculates the cost of a neural network which structure consist on the input layer + 1 hidden layer + 1 output layer
% the cost is calculated only for the output layer (prediction layer)

g = @sigmoid;
obs_q = m = rows(y);
% add bias neurons to X
X = [ones(obs_q, 1) X];

% theta_masked are needed for the regularized cost
theta2_masked = maskTheta(theta2);
theta1_masked = maskTheta(theta1);

% iter trough labels and get sum the cost of each one
% the theta considered for the cost is the one that maps to the output layer (ie. theta 2)
J=0;
for lbl = 1:num_labels
    % lbl : label
    y_lbl = zeros(num_labels,1);
    y_lbl(lbl) = 1;
    % idx : indexes set of rows that match the label
    idx = find(y==lbl);
    a1_lbl = X(idx,:);
    % the j index corresponds to the j-th unit in the previous layer
    % the k index corresponds to the k-th unit in the layer being mapped
    z2_lbl = a1_lbl*theta1';
    a2_lbl = g(z2_lbl);
    % add bias neurons to elements of layer2
    a2_lbl = [ones(rows(a2_lbl), 1) a2_lbl];
    z3_lbl = a2_lbl*theta2';
    hx_lbl = a3_lbl = g(z3_lbl);
    J += sum((log(hx_lbl))*y_lbl+(log(1-(hx_lbl))*(1.-y_lbl)));
end

J = (1/m)*J;% Don't know why the resulting cost is negative, therefore:
J = -J;

% regularization term
% the regularization term considers all thetas that map all layers within the NN.
regularization_cost = (lambda/(2*m))*(sum(sum(theta2_masked.^2)) + sum(sum(theta1_masked.^2)));

% total cost
J = J + regularization_cost;

end