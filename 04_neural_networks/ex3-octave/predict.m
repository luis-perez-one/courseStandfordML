function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

%functions
g=@sigmoid;

% features amd 
obs_q = m = rows(X);

% Add bias neurons to X
X = [ones(m, 1) X];

%num_labels = rows(Theta2); % neurons in last layer

% An = h(Xn)
% layer 1 : A1 = X 
% layer 2 : A2
A2 = h_X2 = g(X*Theta1'); %X*Theta = Z
A3 = h_X3 = g(h_X2*Theta2);

% identify the obtain the index of each row maximum value in A3. That's the selected label for each observation 

max_per_obs = [max_val, col_idx_max_val] = max(A3, [], 2);

p = col_idx_max_val;

end






% You need to return the following variables correctly 

%p = zeros(obs_q, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================



