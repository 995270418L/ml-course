function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% 12 * 1
hthetax = X * theta;
% 
J = 1/ (2*m) * sum((hthetax - y).^2) + lambda / (2*m) * sum(theta(2:end,:).^2);
% J = 1/(2*m) * (h_theta - y)' * (h_theta - y) + ...
%    (lambda/(2*m)) * (theta(2:length(theta)))' * theta(2:length(theta));
theta(1) = 0;
% 1 * 2
grad = 1 / m * ((hthetax - y)' * X)' + lambda / m * theta;






% =========================================================================

grad = grad(:);

end