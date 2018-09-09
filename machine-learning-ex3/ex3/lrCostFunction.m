function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
% 它这边传进来的X 就是 x(i)T 的矩阵集合，所以可以用 X * Theta 来替代 thetaT * X
% 得出来的结果就表示 htheta(x), 即 htheta(x(i))的向量化结果， 得出来了表达式函数，继续计算 J(theta)的向量化
htheta = sigmoid(X * theta);

beta = htheta - y

% 因为 X 是以 x(i)的转置矩阵为列的矩阵
grad = 1/m * X' * beta;
temp = theta;
temp(1) = 0;
grad = grad + lambda / m * temp

J = 1 / m * ( -1 * y' * log(htheta) - (1 - y') * log( 1 - htheta)) + lambda / (2*m) * sum(theta(2:end) .^ 2);

% =============================================================

grad = grad(:);

end
