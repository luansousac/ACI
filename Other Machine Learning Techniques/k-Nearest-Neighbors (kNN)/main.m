%% Setting useful MATLAB Configurations

clc;
close all;
clearvars;
format LONG;

%% Adding important paths into the workspace

addpath('../../Utils/');

% Here you can add the desired dataset. Just be sure to point to the correct
% directory. Example: ../../Datasets/your_directory/
addpath('../../Datasets/vertebral_column_data/');

%% Diplaying some info about the script

fprintf('+----------------------------------------------------------------+\n');
fprintf('|               INTELIGÊNCIA COMPUTACIONAL APLICADA              |\n');
fprintf('|                     K-VIZINHOS MAIS PRÓXIMOS                   |\n');
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
opt = input(['(1) Banknote Annotations, (2) Iris Flower, ' ...
             '(3) Vertebral Column, (4) Artificial Dataset: ']);

switch opt
    case 1
        data = readfile('bank.dat', 5);
    case 2
        [data, classes] = readfile('iris.dat', 4);
        data = labelClasses(data, classes, opt);
    case 3
        [data, classes] = readfile('column_3C.dat', 6);
        data = labelClasses(data, classes, opt);
    case 4
        N = 100;
        v = 0.05;

        M = [0.3  0.3; 0.5  0.8; 0.8  0.5];
        D1 = repmat(M(1,:), N, 1) + v*randn(N,2);
        D2 = repmat(M(2,:), N, 1) + v*randn(N,2);
        D3 = repmat(M(3,:), N, 1) + v*randn(N,2);

        data = [D1 repmat([1 0 0], N, 1);
                D2 repmat([0 1 0], N, 1);
                D3 repmat([0 0 1], N, 1)];
    otherwise
        error('Invalid option!');
end

%% Normalizing data in values between [0,1]

norm_data = normalize(data);

%% Setting important variables

n_iter  = 10;
k_value = 15;

scores     = zeros(n_iter, 1);
avg_scores = zeros(k_value, 1);

%% Performing with different values of 'k'

for k = 1 : k_value
    for i = 1 : n_iter
        % Randomize data in order to have different results
        [X_training, Y_training, X_test, Y_test] = shuffle_data(norm_data);

        % Remove the bias inserted, since kNN is not composed by a neuron.
        X_training(:,1) = [];
        X_test(:,1) = [];

        % Score for a single iteration, i.e., for all the test samples.
        score = 0;

        % Three ways to calculate the euclidean distance:
        for j = 1 : size(X_test, 1)
            % 1st Way
            % data_test   = repmat(X_test(j,:), size(X_training, 1), 1);
            % differences = X_training - data_test;
            % sqrd_diffs  = differences .^ 2;
            % euclid_dist = sqrt(sum(sqrd_diffs, 2));

            % 2nd Way
            % differences = bsxfun(@minus, X_training, X_test(j,:));
            % sqrd_diffs  = differences .^ 2;
            % euclid_dist = sqrt(sum(sqrd_diffs, 2));

            % 3rd Way
            euclid_dist = pdist2(X_training, X_test(j,:));

            % Sorting euclidean distances to get the k smallers 
            [~, inds] = sort(euclid_dist);
            smallest_inds = inds(1:k);
            score = score + (mode(Y_training(smallest_inds)) == Y_test(j));
        end

        perc = (score * 100) / size(X_test,1);
        scores(i) = perc;
    end

    avg_scores(k) = mean(scores);
    fprintf(['With k = %d, the network classified, in average %.2f%% of ', ...
             'the test set correctly.\n'], k, avg_scores(k));
end

%% Displaying summarized results

fprintf('\nSummarizing, the kNN was test with...\n');
fprintf('\tMean accuracy: %.2f\n', mean(avg_scores));
fprintf('\tMinimum accuracy: %.2f\n', min(avg_scores));
fprintf('\tMaximum accuracy: %.2f\n', max(avg_scores));
fprintf('\tStandard Deviation: %.2f\n\n', std(avg_scores));

%% Evaluating the best value for 'k'

[~, max_ind] = max(avg_scores);
fprintf('The best value of "k" was: %.2f\n', max_ind);

%% Ploting results for the report

plot(1:k_value, avg_scores)
xlabel('Valor de k')
ylabel('Acurácia Média')

%% (un)Setting MATLAB Configurations

format;
