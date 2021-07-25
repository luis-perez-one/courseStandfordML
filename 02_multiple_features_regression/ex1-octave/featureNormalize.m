function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
%% ME: YOU DON'T NEED TO. REALLY.
%X_norm = X;
%mu = zeros(1, size(X, 2));
%sigma = zeros(1, size(X, 2));

mu = mean(X,1);
sigma = std(X,1);
X_norm = [];
features = columns(X);
for f = 1:features;
    mu_f = mu(1,f);
    sigma_f = sigma(1,f);
    xf_norm = (X(:,f).-mu_f)./sigma_f;
    X_norm = [X_norm, xf_norm];
endfor

endfunction