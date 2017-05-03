%% Setting useful MATLAB Configurations.

clc;
close all;
clearvars;
format long;

%% Adding important paths into the workspace.

addpath('../../Utils/');
addpath('../../Artificial Neural Networks/Single Layer Perceptron (SLP)/');

% Here you can add the desired dataset. Just be sure to point to the correct
% directory. Example: ../../Datasets/your_directory/
addpath('../../Datasets/haberman_data/');
addpath('../../Datasets/iris_flower_data/');
addpath('../../Datasets/breast_cancer_data/');
addpath('../../Datasets/vertebral_column_data/');
addpath('../../Datasets/banknote_authentication_data/');

%% Displaying some info about the script.

fprintf('+----------------------------------------------------------------+\n');
fprintf('|               INTELIGÊNCIA COMPUTACIONAL APLICADA              |\n');
fprintf('|               PERCEPTRON COM ALGORITMOS GENÉTICOS              |\n');
fprintf('|                 Prof.: Dr. Ajalmar R. Rocha Neto               |\n');
fprintf('|                  Acadêmico: Luan Sousa Cordeiro                |\n');
fprintf('|                              IFCE                              |\n');
fprintf('+----------------------------------------------------------------+\n');

disp(' ');
disp(' ');

%% Importing the data set into the workspace.

disp('Choose the dataset from the options below.');
opt = input(['(1) Banknote Authentication, (2) Iris Flower, ' ...
             '(3) Vertebral Column, (4) Artificial Dataset,\n' ...
             '(5) Breast Cancer, (6) Haberman: ']);

switch opt
    case 1
        data = readfile('bank.dat', 5);
    case 2
        [data, classes] = readfile('iris.dat', 4);
        data = label_classes(data, classes, '2C');
    case 3
        [data, classes] = readfile('column_2C.dat', 6);
        data = label_classes(data, classes, '2C');
    case 4
        N = 100;
        v = 0.05;
        M = [0.3 0.3; 0.5 0.8];

        dat1 = repmat(M(1,:), N, 1) + v*randn(N,2);
        dat2 = repmat(M(2,:), N, 1) + v*randn(N,2);
        data = [dat1 ones(N, 1); dat2 zeros(N, 1)];
    case 5
        data = readfile('breast.dat', 11);
        data(:,11) = data(:,11) / 2 - 1;
    case 6
        data = readfile('haberman.dat', 4);
        data(:,4) = data(:,4) - 1;
    otherwise
        error('Invalid option!');
end

.%% Cleaning useless variables

clear opt;

%% Normalizing data in values between [0,1].

norm_data = normalize(data);

%% Setting important variables.

error = 0;
n_iter = 100;
scores = zeros(n_iter, 1);
population_size = 60;

%% Evolving the network through a Genetic Algorithm.

for i = 1 : n_iter
    %% Shuffling and splitting the data at each iteration.

    % Randomize data in order to have different results
    norm_data = shuffle_data(norm_data);

    % Split data in four sets, as follows.
    [X_training, Y_training, X_test, Y_test] = split_data(norm_data);

    %% Defining a custom fitness function.

    % We need a fitness function for the neural network evolution. The
    % fitness of an individual is its misclassification rate, i.e., the
    % number of errors in the training data. The fitness function only
    % needs the training data (input + classes) to operate.

    % |ga| will call our fitness function with just one argument |W|, but
    % our fitness function has three arguments: |W|, |X_training| and
    % |Y_training|. We can use an anonymous function to capture the values
    % of the additional argument, the input matrices. We create a function
    % handle |FitnessFcn| to an anonymous function that takes one input |W|,
    % but calls |fitness_fnc| with |W|, and the input data. The variables,
    % X_training and Y_training, has a value when the function handle
    % |FitnessFcn| is created, so these values are captured by the anonymous
    % function.
    FitnessFnc = @(W) fitness_fnc(X_training, Y_training, W);

    %% Genetic Algorithm Options Setup.

    options = optimoptions('ga', ...
                           'Display', 'off', ...
                           'UseVectorized', true, ...
                           'MaxGenerations', 500, ...
                           'MaxStallGenerations', 300, ...
                           'ConstraintTolerance', error, ...
                           'PopulationSize', population_size, ...
                           'CrossoverFcn', @crossoverarithmetic);

    %% Running the Genetic Algorithm.

    [W, fval, exitflag, output, population, score] = ...
        ga(FitnessFnc, size(data, 2), [], [], [], [], [], [], [], [], options);

    %% Testing the best set of weights found by the GA.

    scores(i) = test(X_test, Y_test, W)  * 100 / size(X_test, 1);
end

%% Displaying summarized results.

fprintf(['\nSummarizing, the single layer perceptron was tested over %d ' ...
         'iterations (evolution/testing) with...\n'], n_iter);
fprintf('\tMean accuracy: %.2f\n', mean(scores));
fprintf('\tMinimum accuracy: %.2f\n', min(scores));
fprintf('\tMaximum accuracy: %.2f\n', max(scores));
fprintf('\tStandard Deviation: %.2f\n\n', std(scores));

%% (un)Setting MATLAB Configurations.

format;
