%% Setting useful MATLAB Configurations

clc;
close all;
clearvars;
format LONG;

%% Adding important paths into the workspace

addpath('../../Utils/');

% Here you can add the desired dataset. Just be sure to point to the correct
% directory. Example: ../../Datasets/your_directory/
addpath('../../Datasets/');

%% Diplaying some info about the script

fprintf('+----------------------------------------------------------------+\n');
fprintf('|               INTELIGÊNCIA COMPUTACIONAL APLICADA              |\n');
fprintf('|                   REDE PERCEPTRON MULTICAMADA                  |\n');
fprintf('|                 Prof.: Dr. Ajalmar R. Rocha Neto               |\n');
fprintf('|                  Acadêmico: Luan Sousa Cordeiro                |\n');
fprintf('|                              IFCE                              |\n');
fprintf('+----------------------------------------------------------------+\n');

disp(' ');
disp(' ');
disp('Press any key to continue...');
pause

%% Importing the data set into the workspace

disp('Choose the dataset from the options below.');
opt = input(['(1) Banknote Authentication, (2) Iris Flower, ' ...
             '(3) Vertebral Column, (4) Artificial Dataset: ']);

switch opt
    case 1
        data = readfile('bank.dat', 5);
        classes = [0; 1];
    case 2
        [data, classes] = readfile('iris.dat', 4);
        data = label_classes(data, classes);
    case 3
        [data, classes] = readfile('column_3C.dat', 6);
        data = label_classes(data, classes);
    case 4
        N = 100;
        v = 0.05;
        M = [0.3 0.3; 0.5 0.8; 0.8 0.5];
        classes = [1 0 0; 0 1 0; 0 0 1];

        dat1 = repmat(M(1,:), N, 1) + v*randn(N,2);
        dat2 = repmat(M(2,:), N, 1) + v*randn(N,2);
        dat3 = repmat(M(3,:), N, 1) + v*randn(N,2);
        data = [dat1 repmat(classes(1,:), N, 1);
                dat2 repmat(classes(2,:), N, 1);
                dat3 repmat(classes(3,:), N, 1)];
    otherwise
        error('Invalid option!');
end

%% Normalizing data in values between [0,1]

%norm_data = normalize(data);
n_features = size(data, 2);
norm_data = [normalize(data(:,1:n_features-1)) data(:,n_features)];

%% Setting important variables

% Iteration control
n_iter = 10;
scores = zeros(n_iter, 1);

% Training settings
err = 1e-5;
l_rate = 0.05;
n_epochs = 2500;

if iscell(classes)
    n_classes = length(unique(classes));
else
    n_classes = length(unique(classes, 'rows'));
end

if n_classes < 3
    n_neurons = 1;
else
    n_neurons = n_classes;
end

%% K-fold cross-validation with grid-search

    % Paramenters to the grid-search.
    n_folds = 10;
    neurons = 11:15;
    l_rate  = 0.05:-0.005:0.03;

    % Randomize data in order to have different results.
    norm_data = shuffle_data(norm_data);

    % Split the data in four sets, as follows.
    [X_training, Y_training, X_test, Y_test] = split_data(norm_data, n_neurons);
%{
    % "Split" the training data into k-folds, by getting its indices.
    n_elements = floor(length(X_training) / n_folds);
    grid_array = zeros(length(neurons), length(l_rate));

    %fprintf('\n%s Validation results %s\n\n', repmat('_', 42, 1), repmat('_', 42, 1));

    % The following nested loop perform the grid search for the best parameters for the network.
    for j = 1 : length(neurons)
        for k = 1 : length(l_rate)
            accuracy = zeros(n_folds, 1);

            for m = 1 : n_folds
                v_inds = (m-1) * n_elements + 1 : n_elements * m;
                t_inds = setdiff(1 : length(X_training), v_inds);

                % Training the network with (k-1) folds.
                [W1, W2, errorMatrix] = train(X_training, Y_training, l_rate(k), n_epochs, err, neurons(j));

                % Validating the network for this specific set of parameters and k-th fold.
                %val_right = 0;
                %val_wrong = zeros(size(X_test));
                fprintf('With %d neurons and sigma = %.2f, ', neurons(j), l_rate(k));

                acc = test(X_training(v_inds,:), Y_training(v_inds,:), W1, W2);
                accuracy(m) = (acc * 100) / size(X_training(v_inds,:), 1);

                fprintf('the network classified %.2f%% of the validation set correctly.\n', acc);
            end

            %disp(' ');
            grid_array(j,k) = mean(accuracy(m));
        end
    end

    fprintf('%s\n', repmat('_', 104, 1));

    % Grid searching for the best set of parameters.
    max_accuracy = max(grid_array(:));
    [x,y] = ind2sub(size(grid_array), find(grid_array == max_accuracy, 1, 'last'));
%}
%% Training/Testing/Results iteration

for i = 1 : n_iter
%% Training the network with the all training data and the selected parameters.

    % Randomize data in order to have different results
    norm_data = shuffle_data(norm_data);

    % Split the data in four sets, as follows.
    [X_training, Y_training, X_test, Y_test] = split_data(norm_data, n_neurons);

    % Training the network
    % [W1, W2, errorMatrix] = train(X_training, Y_training, l_rate(1), n_epochs, err, neurons(x));
    [W1, W2, errorMatrix] = train(X_training, Y_training, 0.01, n_epochs, err, 12);

%% Testing the best model generated by the network

    score = test(X_test, Y_test, W1, W2);
    scores(i) = (score * 100) / size(X_test, 1);

    % Displaying the accuracy for each iteration
    fprintf('The network classified %.2f%% of the test set correctly.\n', scores(i));

%% Ploting results for the report    

    % Ploting the error matrix
    % plotarErros(errorMatrix);

    if opt == 4
        C1 = [];
        C2 = [];
        C3 = [];
        j = 1;
        l = 1;
        k = 1;

        for x1 = 0:0.003:1
            for x2 = 0:0.003:1
                h = W1 * [-1; x1; x2];
                y = [1; logsig(h)];

                o = logsig(W2 * y);
                o = evaluate(o, 2);
                p = [x1 x2];

                % Storing for plot purposes
                if isequal(o', [1 0 0])
                    C1(j,:) = p;
                    j = j + 1;
                elseif isequal(o', [0 1 0])
                    C2(l,:) = p;
                    l = l + 1;
                else
                    C3(k,:) = p;
                    k = k + 1;
                end
            end
        end

        figure
        hold on
        axis([0 1 0 1])
        plot(C1(:,1), C1(:,2), '.', 'Color', [0.95 0.87 0.73], 'LineStyle', '-', ...
                                    'LineWidth', 3.0, 'HandleVisibility', 'off')
        plot(C2(:,1), C2(:,2), '.', 'Color', [0.80 0.88 0.97], 'LineStyle', '-', ...
                                    'LineWidth', 3.0, 'HandleVisibility', 'off')
        plot(C3(:,1), C3(:,2), '.', 'Color', [0.80 0.80 0.80], 'LineStyle', '-', ...
                                    'LineWidth', 3.0, 'HandleVisibility', 'off')

        plot(dat1(:,1), dat1(:,2), '*', 'Color', [0.60 0.20 0.00], ...
                                        'DisplayName', 'Classe 1');
        plot(dat2(:,1), dat2(:,2), '*', 'Color', [0.00 0.45 0.74], ...
                                        'DisplayName', 'Classe 2');
        plot(dat3(:,1), dat3(:,2), '*', 'Color', [0.31 0.31 0.31], ...
                                        'DisplayName', 'Classe 3');

        legend show
    end

    pause(0.1);
end

%% Displaying summarized results

fprintf(['\nSummarizing, the multilayer perceptron was tested over %d ' ...
         'iterations (training/testing) with...\n'], n_iter);
fprintf('\tMean accuracy: %.2f\n', mean(scores));
fprintf('\tMinimum accuracy: %.2f\n', min(scores));
fprintf('\tMaximum accuracy: %.2f\n', max(scores));
fprintf('\tStandard Deviation: %.2f\n\n', std(scores));

%% (un)Setting MATLAB Configurations

format;
