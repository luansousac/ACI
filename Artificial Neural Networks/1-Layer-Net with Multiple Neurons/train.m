function [ W ] = train( X_training, Y_training, W, l_rate, n_epochs )
%TRAIN train the 1-layer with mutiple perceptron neurons
%
%   Training each sample from the training data to update the weight
%   matrix iteratively in the inside loop

    n_patterns = size(X_training, 1);

    for j = 1 : n_epochs
        for k = 1 : n_patterns
            % Computing the output 'Y' using the logistic function
            Y = logsig(W * X_training(k,:)');

            % Computing the error
            pattern_error = Y_training(k,:) - Y';

            % Updating the weight matrix
            W = W + (l_rate * pattern_error' * X_training(k,:));
        end
    end

end
