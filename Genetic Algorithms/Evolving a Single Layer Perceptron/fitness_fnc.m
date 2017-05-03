function [ scores ] = fitness_fnc( X, Y, W )
%FITNESS_FUNCTION stores the fitness from all individuals in the population
%
%   This custom fitness function uses the perceptron step function to check
%   the correctness of each individual (set of weights), i.e., whether it
%   classifies the data in a correct way. It uses the training data to
%   simulate the perceptron training algorithm, that finds, iteratively,
%   the best set of weights that fits best to the input data.

    % The size of the population and the number of training samples.
    n_individuals = length(W);
    n_patterns = length(X);

    % Otimizing the script by declaring the output vector outside the loop.
    scores = zeros(n_individuals, 1);

    % Testing the weights (each individual) on the training samples.
    for j = 1 : n_individuals

        % Computing the output 'Y' using the step function.
        y = (W(j,:) * X' >= 0);

		% Stores the error of each individual (i.e., the number of
		% misclassifications).
        scores(j) = n_patterns - sum(y == Y');

    end

end
