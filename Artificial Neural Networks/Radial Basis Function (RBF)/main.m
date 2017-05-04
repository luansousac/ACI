%% Setting useful MATLAB Configurations

clc;
close all;
clearvars;
format long;

%% Adding important paths into the workspace

addpath('../../Utils/');

% Here you can add the desired dataset. Just be sure to point to the correct
% directory. Example: ../../Datasets/your_directory/
addpath('../../Datasets/vertebral_column_data/');
addpath('../../Datasets/iris_flower_data/');
addpath('../../Datasets/breast_cancer_data/');
addpath('../../Datasets/haberman_data/');
addpath('../../Datasets/wine_data/');
addpath('../../Datasets/banknote_authentication_data/');
addpath('../../Datasets/letter_data/');

%% Displaying some info about the script

fprintf('+----------------------------------------------------------------+\n');
fprintf('|               INTELIGÊNCIA COMPUTACIONAL APLICADA              |\n');
fprintf('|                 REDE FUNÇÃO DE BASE RADIAL (RBF)               |\n');
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
opt = input(['(1) Banknote Authentication, (2) Iris Flower, (3) Vertebral Column, ' ...
             '(4) Artificial Dataset: ']);

switch opt
    case 1
        classes = [1; 2];
        data = readfile('bank.dat', 5);
        data(:,5) = data(:,5) + 1;
    case 2
        [data, classes] = readfile('iris.dat', 4);
        data = label_classes_num(data, classes);
    case 3
        [data, classes] = readfile('column_3C.dat', 6);
        data = label_classes_num(data, classes);
    case 4
        classes = [1; 2];
        data = readfile('breast.dat', 10);
        data(:,10) = data(:,10) / 2;
    case 5
        [data, classes] = readfile('haberman.dat', 3);
        data = label_classes_num(data, classes);
    case 6
        data = readfile('wine.dat', 13);
        data(:,14) = data(:,1);
        data(:,1) = [];
        classes = unique(data(:,13));
    case 7
        data = readfile('letter-recognition.dat', 17);
        data(:,18) = data(:,1);
        data(:,1) = [];
        classes = unique(data(:,17));
    case 8
        N = 100;
        v = 0.05;
        M = [0.3 0.3; 0.5 0.8; 0.8 0.5];
        classes = [1; 2; 3];

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

n_features = size(data, 2);
norm_data = [normalize(data(:,1:n_features-1)) data(:,n_features)];

%% Setting important variables

% Iteration control
n_iter = 50;
accuracies = zeros(n_iter,1);

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
    neurons = 5:25;
    sigmas  = 0.01:0.001:0.02;
    %sigmas  = 3:1:5;
    %sigmas  = 5:1:8;
    %sigmas(sigmas == 0) = [];
    %sigmas  = 1 ./ (2.*(1:0.5:2).^2);
    %sigmas  = (0.4-0.1) * rand(5,1) + 0.1;

    % Randomize data in order to have different results.
    norm_data = shuffle_data(norm_data);

    % Split the data in four sets, as follows.
    [X_training, Y_training, X_test, Y_test] = split_data(norm_data);

    % "Split" the training data into k-folds, by getting its indices.
    n_elements = floor(length(X_training) / n_folds);
    grid_array = zeros(length(neurons), length(sigmas));

    %fprintf('\n%s Validation results %s\n\n', repmat('_', 42, 1), repmat('_', 42, 1));

    % The following nested loop perform the grid search for the best parameters for the network.
    for j = 1 : length(neurons)
        for k = 1 : length(sigmas)
            accuracy = zeros(n_folds, 1);

            for m = 1 : n_folds
                v_inds = (m-1) * n_elements + 1 : n_elements * m;
                t_inds = setdiff(1 : length(X_training), v_inds);

                % Training the network with (k-1) folds.
                [centers, betas, theta] = train(X_training(t_inds,:), Y_training(t_inds,:), ...
                    neurons(j), sigmas(k));

                % Validating the network for this specific set of parameters and k-th fold.
                val_right = 0;
                val_wrong = zeros(size(X_test));
                %fprintf('With %d neurons and sigma = %.2f, ', neurons(j), sigmas(k));

                for n = 1 : size(v_inds, 2)
                    scores = test(X_training(v_inds(n), :), centers, betas, theta);
                    [~, category] = max(scores);

                    if category == Y_training(v_inds(n))
                        val_right = val_right + 1;
                    else
                        val_wrong(n,:) = X_training(v_inds(n), :);
                    end
                end

                val_wrong(~any(val_wrong,2), :) = [];

                acc = val_right / size(v_inds, 2) * 100;
                accuracy(m) = acc;
                %fprintf('the network classified %.2f%% of the validation set correctly.\n', acc);
            end

            %disp(' ');
            grid_array(j,k) = mean(accuracy(m));
        end
    end

    fprintf('%s\n', repmat('_', 104, 1));

    % Grid searching for the best set of parameters.
    max_accuracy = max(grid_array(:));
    [x,y] = ind2sub(size(grid_array), find(grid_array == max_accuracy, 1, 'last'));

%% Training/Validating/Testing/Results iteration

for i = 1 : n_iter
%% Training the network with the all training data and the selected parameters.

    norm_data = shuffle_data(norm_data);

    % Split the data in four sets, as follows.
    [X_training, Y_training, X_test, Y_test] = split_data(norm_data);

    [centers, betas, theta] = train(X_training, Y_training, neurons(x), sigmas(y));

%% Testing the best model generated by the network

    right = 0;
    wrong = zeros(size(X_test));

    for j = 1 : size(X_test, 1);
        scores = test(X_test(j, :), centers, betas, theta);
        [~, category] = max(scores);

        if category == Y_test(j)
            right = right + 1;
        else
            wrong(j,:) = X_test(j, :);
        end
    end

    wrong(~any(wrong,2), :) = [];

    acc = right / size(X_test, 1) * 100;
    accuracies(i) = acc;

    % Displaying the accuracy for each iteration
    %fprintf('\n%s Testing results %s\n', repmat('_', 44, 1), repmat('_', 43, 1));
    %fprintf('\nParameters:\n\tNumber of hidden neurons = %d.\n\tSigma = %.2f.\n', neurons(x), sigmas(y));
    %fprintf('\nThe network classified %.2f%% of the test set correctly.\n', acc);
    %fprintf('%s\n\n\n', repmat('_', 104, 1));

%% Ploting results for the report    

    % Ploting the error matrix
    % plotarErros(errorMatrix);

    if opt == 8
%{
        C1 = [];
        C2 = [];
        C3 = [];
        j = 1;
        l = 1;
        k = 1;

        for x1 = 0:0.003:1
            for x2 = 0:0.003:1
                h = theta * [x1; x2];
                y = [1; logsig(h)];

                o = logsig(W2 * y);
                o = evaluate(o, 2);
                p = [x1 x2];

                % Storing for plot purposes
                if isequal(o', 1)
                    C1(j,:) = p;
                    j = j + 1;
                elseif isequal(o', 2)
                    C2(l,:) = p;
                    l = l + 1;
                else
                    C3(k,:) = p;
                    k = k + 1;
                end
            end
        end
%}
        figure
        %hold on
        %axis([0 1 0 1])
%         plot(C1(:,1), C1(:,2), '.', 'Color', [0.95 0.87 0.73], 'LineStyle', '-', ...
%                                     'LineWidth', 3.0, 'HandleVisibility', 'off')
%         plot(C2(:,1), C2(:,2), '.', 'Color', [0.80 0.88 0.97], 'LineStyle', '-', ...
%                                     'LineWidth', 3.0, 'HandleVisibility', 'off')
%         plot(C3(:,1), C3(:,2), '.', 'Color', [0.80 0.80 0.80], 'LineStyle', '-', ...
%                                     'LineWidth', 3.0, 'HandleVisibility', 'off')

        contour(neurons, sigmas, grid_array')
%         plot(norm_data(norm_data(:,3) == 1,1), norm_data(norm_data(:,3) == 1,2), '*', 'Color', [0.60 0.20 0.00], ...
%                                         'DisplayName', 'Classe 1');
%         plot(norm_data(norm_data(:,3) == 2,1), norm_data(norm_data(:,3) == 2,2), '*', 'Color', [0.00 0.45 0.74], ...
%                                         'DisplayName', 'Classe 2');
%         plot(norm_data(norm_data(:,3) == 3,1), norm_data(norm_data(:,3) == 3,2), '*', 'Color', [0.31 0.31 0.31], ...
%                                         'DisplayName', 'Classe 3');

        %legend show
    end

    %pause(0.1);
end

%% Displaying summarized results

fprintf('\nSummarizing, the radial basis function network was tested with...\n');
fprintf('\tMean accuracy: %.2f\n', mean(accuracies));
fprintf('\tMinimum accuracy: %.2f\n', min(accuracies));
fprintf('\tMaximum accuracy: %.2f\n', max(accuracies));
fprintf('\tStandard Deviation: %.2f\n\n', std(accuracies));

%% (un)Setting MATLAB Configurations

format;
