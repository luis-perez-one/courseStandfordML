% function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y



% Initialize some useful values
% m = length(y); % number of training examples
%% I will define this variable under the relevant functions

% You need to return the following variables correctly 
% J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

function J = computeCost(X, y, theta)

    J = 0;

    function validate_theta_as_vector(theta)
        vector = (size(theta)(1)>1) && (size(theta)(2)==1);
        error_mssg = "theta is not a vector with at least 2 values!. Note: an intersect on the y-axis is considered.";
        assert(vector, error_mssg);
    endfunction


    function check_obs_vs_responses_qty(X, y)
        y_is_vector = (size(y)(1)>1) && (size(y)(2)==1);
        error_mssg = "y is not a valid vector!";
        assert(y_is_vector == true, error_mssg);
        obs_qty = rows(X);
        responses_qty = rows(y);
        error_mssg = "the observations (rows in X) and responses (rows in y) quantities are different!";
        assert(obs_qty == responses_qty, error_mssg);
    endfunction


    function check_coef_vs_features_qty(X, theta)
        coef_qty = rows(theta);
        features_qty = columns(X);
        error_mssg = "the coefficients (rows in theta) and features (columns in X) quantities are different!";
        assert(coef_qty == features_qty, error_mssg);
    endfunction

    function X = add_theta0_column(X)
        %theta0 would be the intersect if added
        m = rows(X);
        X = [ones(m,1), X];
    endfunction


    function y_prediction = evalHypothesis(X, theta)
        y_prediction = [];
        for i = 1:rows(X);  %i: i-th observation of the data set
            x = X(i,:);
            y_prediction_i = x*theta;
            y_prediction = [y_prediction; y_prediction_i];
        endfor
    endfunction
    
    %X = add_theta0_column(X)
    validate_theta_as_vector(theta);
    check_obs_vs_responses_qty(X, y);
    check_coef_vs_features_qty(X, theta);
    y_prediction = evalHypothesis(X, theta);
    error = y_prediction - y;
    m = length(y);
    J = (1/(2*m))*(sum(error.^2));

endfunction
% =========================================================================

%end
