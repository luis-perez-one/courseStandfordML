function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% my own functions
    function X = add_theta0_column(X)
        %theta0 would be the intersect if added
        m = rows(X);
        X = [ones(m,1), X];
    endfunction


    function error = calc_error(X, y, theta)
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


        function y_prediction = evalHypothesis(X, theta)
            y_prediction = [];
            for i = 1:rows(X);  %i: i-th observation of the data set
                x = X(i,:);
                y_prediction_i = dot(theta,x);
                y_prediction = [y_prediction; y_prediction_i];
            endfor
        endfunction
        
        check_obs_vs_responses_qty(X, y);
        check_coef_vs_features_qty(X, theta);
        y_prediction = evalHypothesis(X, theta);
        error = y_prediction - y;
    endfunction


    function plot_J_history(J_history)
        num_iters = length(J_history);
        plot(1:num_iters, J_history)
        hold on
        ylabel('J(theta)')
        title('Gradient descendent performance')
        hold off
    endfunction


    %X = add_theta0_column(X);
    m = length(y); % number of training examples
    features_qty = columns(X); % number of features 
    alpha_m_ratio = alpha/m;
    old_cost = computeCostMulti(X, y, theta);
    new_theta = [];
    J_history = [];
    J_difference_history = [];
    
    iters_to_go = num_iters;
    while iters_to_go != 0;
        error = calc_error(X, y, theta);
        for f = 1:features_qty;
            new_theta_f = theta(f,1) - alpha_m_ratio*sum(X(:,f).*error);
            new_theta = [new_theta; new_theta_f];
        endfor
        theta = new_theta;
        new_theta = [];
        cost = computeCostMulti(X, y, theta);
        J_history = [J_history; cost];
        J_difference = cost - old_cost;
        J_difference_history = [J_difference_history; J_difference];
        old_J = cost;
        iters_to_go -= 1;

    endwhile

    #plot_J_history(J_history)

endfunction
