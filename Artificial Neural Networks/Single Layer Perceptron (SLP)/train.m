function [ W ] = train( X_training, Y_training, W, l_rate, n_epochs )
%TRAIN train the 1-layer with mutiple perceptron neurons
%
%   Training each sample from the training data to update the weight
%   vector iteratively in the inside loop

    n_patterns = size(X_training, 1);

    for j = 1 : n_epochs
        for k = 1 : n_patterns
            % Computing the output 'Y' using the step function
            Y = (W * X_training(k,:)' >= 0);

            % Computing the error
            error = Y_training(k) - Y;

            % Updating the weight vector
            W = W + (l_rate * error * X_training(k,:));
        end
    end

end

