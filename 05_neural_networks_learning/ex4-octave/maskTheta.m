function [masked_theta] = maskTheta(theta)
% returns a masked theta so the weight of the bias unit (neuron) is not regularized when calculated the cost
mask = ones(size(theta));
if (columns(theta) == 1)
    % theta is a vector
    mask(1) = 0;
else
    % theta is a matrix
    mask(:,1) = 0;
endif
masked_theta = theta .* mask;

end