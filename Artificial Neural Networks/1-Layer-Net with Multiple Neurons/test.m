function [ score ] = test( X_test, Y_test, W )
%TEST test the model generated in the training section
%
%   Testing each sample from the test data in order to see if the model
%   classifies the data correctly.

    score = 0;
    n_neurons = size(Y_test, 2);
    n_patterns = size(X_test, 1);

    % Otimizing the script by declaring the output vector outside the loop
    Y = zeros(n_neurons, n_patterns);

    % Testing the model for each test sample
    for j = 1 : n_patterns
        % Computing the output 'Y' using the logistic function
        Y(:,j) = logsig(W * X_test(j,:)');

        % Binarizing each activation from the output 'Y'
        Y(:,j) = evaluate(Y(:,j), 2);

        % Increments the score if the classification is correct
        score = score + isequal(Y(:,j)', Y_test(j,:));
    end

end
