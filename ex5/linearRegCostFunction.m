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
%common part
h=X*theta;
error=h-y;
%regularisation squared error
error_sqr=error.^2;
q=sum(error_sqr);
lr=(1/(2*m))*q;
%regularisation cost function
copy_theta=theta;
copy_theta(1)=0;
reg=(lambda/(2*m))*sum(copy_theta.^2);
J=lr+reg;
%regularisation gradient
l=X'*error;
gr=(1/m)*(l);
regT=(lambda/m)*theta;
regT(1)=0;
grad=gr+regT;
% =========================================================================

grad = grad(:);

end
