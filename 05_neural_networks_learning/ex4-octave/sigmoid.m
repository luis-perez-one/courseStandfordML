%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

function g = sigmoid(z)
f = inline("1/(1+e^(-x))");
g = arrayfun(f, z);
end