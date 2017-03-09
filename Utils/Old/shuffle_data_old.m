function [ X_training, Y_training, X_test, Y_test ] = shuffle_data( data, neurons )
%SHUFFLE_DATA shuffles the presented data set in order to achieve different
% results at each execution. This is a general function, working for any
% data set. The function is splitted in three parts:
%       1. Generates random indices (integers) from 1 to number of patterns
%		   (samples, rows) from the data set.
%       2. Uses these random indices to filter the original data and split
%		   it into two sets: training and test.
%       3. Inserts the bias (column vector) onto the two sets above.

    if nargin < 2
        neurons = 1;
    end
    n_patterns = size(data, 1);
    n_features = size(data, 2);
    down_bound = n_features - neurons;

    % Setting the size of the train set. The trial (test) set is defined
    % by 100 - train_percent.
    train_percent = 80;

    % Getting random indices
    index = randperm(n_patterns);
    bound = floor(n_patterns * train_percent/100);

    % Selecting randomized indices to get random data
    train_set = index(1 : bound);
    trial_set = index(bound + 1 : n_patterns);

    % Picking random data for training purposes through random indices
    X_training = data(train_set, 1 : down_bound);
    Y_training = data(train_set, down_bound + 1 : n_features);

    % Picking random data for test purposes through random indices
    X_test = data(trial_set, 1 : down_bound);
    Y_test = data(trial_set, down_bound + 1 : n_features);

    % Inserting the bias for each training sample
    bias = -1 * ones(bound,1);
    X_training = [bias X_training];

    % Inserting the bias for each testing sample
    bias = -1 * ones(n_patterns - bound, 1);
    X_test = [bias X_test];

end
