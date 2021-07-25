function [J grad] = nnCostFunction(nn_params, input_layer_size hidden_layer_size, num_labels, X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

g=@sigmoid;
m = rows(y);

% add bias neurons to X
X = [ones(m, 1) X];
% initialize cost
J = 0;

theta = Theta1;
h_xk = g(X*theta');
neurons = rows(theta);
for k = 1:neurons
    % cost without regularization
    yk = zeros(neurons,1);
    yk(k) = 1;
    J += (-1/m)*(sum((log(h_xk))*yk+(log(1-(h_xk))*(1.-yk))));
end
% regularization
theta_mask = ones(size(theta));
theta_mask(:,1) = 0;
theta_masked = theta .* theta_mask;
J += (lambda/(2*m))*sum((sum(theta_masked.^2)));


theta = Theta2;
h_xk = [ones(m, 1) h_xk];
h_xk = g(h_xk*theta');
neurons = rows(theta);
for k = 1:neurons
    % cost without regularization
    yk = zeros(neurons,1);
    yk(k) = 1;
    J += (-1/m)*(sum((log(h_xk))*yk+(log(1-(h_xk))*(1.-yk))));
end
% regularization
theta_mask = ones(size(theta));
theta_mask(:,1) = 0;
theta_masked = theta .* theta_mask;
J += (lambda/(2*m))*sum((sum(theta_masked.^2)));







% unrolled theta (aka. theta vector)
theta = [Theta1(:); Theta2(:);];







         
% You need to return the following variables correctly 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));






% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
