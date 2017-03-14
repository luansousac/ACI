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
fprintf('|            REDE COM MEMÓRIA ASSOCIATIVA LINEAR ÓTIMA           |\n');
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

norm_data = normalize(data);

%% Setting important variables

% Iteration control
n_iter = 10;
scores = zeros(n_iter, 1);

% Training settings
n_classes = length(unique(classes,'rows'));

if n_classes < 3
    n_neurons = 1;
else
    n_neurons = n_classes;
end

%% Training/Testing/Results iteration

for i = 1 : n_iter
%% Training the network

    % Randomize data in order to have different results
    norm_data = shuffle_data(norm_data);

    % Split data in four sets, as follows.
    [X_training, Y_training, X_test, Y_test] = split_data(norm_data, n_neurons);

    % Training in batch mode (OLAM). The default calculation is:
    % W = inv(X_training' * X_training) * (X_training' * Y_training).
    % But, in order to optimize the code, it becomes the following
    % calculation, which is similar to the above.
    W = (X_training' * X_training) \ (X_training' * Y_training);

%% Testing the associative memory

    % Adjusting the weight vector found during the training phase so
    % the output can be approximately in the interval [-a, a].
    W = W - 0.5;
    Y = X_test * W;

    % Binarizing the activations from the output 'Y'
    Y = evaluate(Y');

    % Getting the hit rate of the network
    score = sum(Y_test == Y', 2);
    score = sum(score == n_neurons);
    scores(i) = (score * 100) / size(X_test, 1);

    % Displaying the accuracy for each iteration
    fprintf('The network classified %.2f%% of the test set correctly.\n', scores(i));

%% Ploting results for the report

    % hold on
    % plot(1 : n_iter, scores, 'b')
    % xlabel('Iterações')
    % ylabel('Acurácia')
    %
    % Create the data for the standard deviations and datasets
    % stds = [0.000 1.207 7.070];
    % dataset = {'Banknote Authentication', 'Iris Flower', 'Vertebral Column'};
    % 
    % Plot the standard deviations on a horizontal bar chart
    % figure
    % bar(stds)
    % 
    % Change the Y axis tick labels to use the datasets
    % set(gca, 'XTick', 1:3)
    % set(gca, 'XTickLabel', dataset)

    if opt == 4
        C1 = [];
        C2 = [];
        C3 = [];
        j = 1;
        l = 1;
        k = 1;

        for x1 = 0:0.003:1
            for x2 = 0:0.003:1
                u = W*[-1; x1; x2];
                u = logsig(u);
                u = evaluate(u);
                p = [x1 x2];

                % Storing for plot purposes
                if isequal(u', [1 0 0])
                    C1(j,:) = p;
                    j = j + 1;
                elseif isequal(u', [0 1 0])
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

    pause(0.1)
end

%% Displaying summarized results

fprintf('\nSummarizing, the 1-layer network with OLAM was tested with...\n');
fprintf('\tMean accuracy: %.2f\n', mean(scores));
fprintf('\tMinimum accuracy: %.2f\n', min(scores));
fprintf('\tMaximum accuracy: %.2f\n', max(scores));
fprintf('\tStandard Deviation: %.2f\n\n', std(scores));

%% (un)Setting MATLAB Configurations

format;
