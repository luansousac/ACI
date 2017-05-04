function [ W1, W2, error_matrix ] = train( X, Y, l_rate, n_epochs, error, n_hidden )
%TRAIN train the multilayer perceptron with the backpropagation algorithm
%   Here, it is performed the network training in order to learn the weight
%   vector using the traditional backpropagation algorithm with a decreasing
%   learning rate for a quick convergence. This is a general function, which
%   works for any data set.

    % Properties of the MLP network: number of hidden layers, number of hidden
    % neurons per layer and number of neurons on the output layer.
    %n_hidden_neurons = 6;
    n_hidden_neurons = n_hidden;
    n_output_neurons = size(Y, 2);

    % Dimensions of the data.
    n_patterns = size(X, 1);
    n_features = size(X, 2);

    % Minimum and maximum learning rates of the network.
    min_l_rate = 0.01;
    max_l_rate = l_rate;
    t = 1:2500; T = 2500;
    %n_l_rate = repmat(l_rate, 1, 2500) .* (max_l_rate*(min_l_rate/max_l_rate).^(t./T));
    n_l_rate = repmat(l_rate, 1, 2500) .* exp(-t./T);

    % Initiating random weight matrices, W1 and W2. These weight matrices
    % refer to the weights applied on each neuron on the hidden and output
    % layers, respectively.
    W1 = initWeights(n_hidden_neurons, n_features, -1, 1);
    W2 = initWeights(n_output_neurons, n_hidden_neurons+1, -1, 1);

    % Control variables for each training epoch
    j = 0;
    epoch = 0;
    epoch_error = 1;
    error_reached = 0;
    error_matrix = zeros(1, n_epochs);

    % Backpropagation algorithm.
    while ~error_reached && j < n_epochs
        prevEpochError = epoch_error;
        epoch_error = 0;

        % Internal loop for training each pattern sequentially.
        for i = 1 : n_patterns
            %% Forward pass

            % First, the neurons activations on the hidden layer are calculated.
            % Next, it is applied a non-linear function (logistic, namely) over
            % these activations, which will be the input for the output layer.
            % In the end, a bias is added to the hidden layer's output.
            hiddenlayer_act = W1 * X(i,:)';
            Y1 = [1; logsig(hiddenlayer_act)];

            % First, the neurons activations on the output layer are calculated.
            % Next, it is applied a non-linear function (logistic, namely) over
            % these activations, which will serve to calculate the output error,
            % and to the backward pass of the backpropagation algorithm.
            outputlayerInput = W2 * Y1;
            Y2 = logsig(outputlayerInput);

            %% Backward pass

            % Calculate the error of the ouput layer.
            out_error = Y(i,:) - Y2';

            % Calculate the derivative from the output layer and then calculate
            % the output gradient, using the traditional delta rule.
            output_gradient = out_error .* actFunctionDrv(outputlayerInput');

            % Adjusting the output weight matrix.
            Waux = zeros(n_output_neurons, n_hidden_neurons+1);
            for idc = 1 : n_output_neurons
                Waux(idc,:) = (l_rate .* output_gradient(idc) .* Y1);
                W2(idc,:) = W2(idc,:) + Waux(idc,:);
            end

            % Calculate the derivative from the hidden layer and then calculate
            % the hidden gradient, using the generalized delta rule.
            hiddden_gradient = (output_gradient * W2(:,2:size(Y1,1))) .* ...
                                actFunctionDrv(hiddenlayer_act');

            % Adjusting the hidden weight matrix.
            Waux = zeros(n_hidden_neurons, n_features);
            for idc = 1 : n_hidden_neurons
                Waux(idc,:) = (l_rate .* hiddden_gradient(idc) .* X(i,:));
                W1(idc,:) = W1(idc,:) + Waux(idc,:);
            end

            % Adjusting the current epoch error.
            epoch_error = epoch_error + sum(out_error .^ 2) / 2;
        end

        % Checking whether the error limit was reached.
        error_reached = abs(epoch_error - prevEpochError) < error;

        % Storing the epoch error in the error_matrix.
        error_matrix(:, epoch + 1) = epoch_error / n_patterns;
        epoch = epoch + 1;
        j = j + 1;

        % Adjusting the decreasing learning rate.
        %new_l_rate = max_l_rate / (1 + j/n_epochs);
        l_rate = max(min_l_rate, n_l_rate(j));
    end

end
