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

%% Diplaying some info about the script

fprintf('+----------------------------------------------------------------+\n');
fprintf('|               INTELIGÊNCIA COMPUTACIONAL APLICADA              |\n');
fprintf('|              MÁQUINAS DE APRENDIZADO EXTREMO (ELM)             |\n');
fprintf('|                 Prof.: Dr. Ajalmar R. Rocha Neto               |\n');
fprintf('|                  Acadêmico: Luan Sousa Cordeiro                |\n');
fprintf('|                              IFCE                              |\n');
fprintf('+----------------------------------------------------------------+\n');

disp(' ');
disp(' ');

%% Generating the data

% Gerando base X
X = 0.01:0.01:10;
Y = sin(2*X);

randIndexes = randperm(length(X));

% Adicionando Rúido ao seno
noise = 0.1*randn(1, length(Y)) - 0.1/2;
Y = Y + noise;

%% Splitting the data

X_training = X(randIndexes(1:floor(length(X) * 0.8)));
Y_training = Y(randIndexes(1:floor(length(Y) * 0.8)));

X_test = X(randIndexes(floor(length(X) * 0.8) + 1:end));
Y_test = Y(randIndexes(floor(length(Y) * 0.8) + 1:end));

%% Setting important variables

n_tests = 1;
n_folds = 5;
mse_vector = zeros;

data = [X_training ; Y_training];
s_fold = length(X_training) / n_folds;

for i = 1 : 5
    fold(i,:,:) = data(:, s_fold*(i-1)+1 : s_fold*i);
end

%% k-fold cross-validation

for k = 1 : 100    
    mse_fold = 0;

    for j = 1 : n_folds
        %% Using the chosen folds to train the network
        xTestFold = [];
        yTestFold = [];
        yFold = [];
        xFold = [];
        
        for i = 1 : n_folds
            for l = 1 : s_fold
                if i ~= j
                    xFold = [xFold fold(i,1,l)];
                    yFold = [yFold fold(i,2,l)];
                else
                    xTestFold = [xTestFold fold(i,1,l)];
                    yTestFold = [yTestFold fold(i,2,l)];
                end
            end
        end

        %% Training the network

        hidden_weights = rand(k, 1);
        hidden_outputs = sigmf(hidden_weights * xFold, [1 0]);
        
        %% Testing the network

        H = sigmf(hidden_weights * xTestFold, [1 0]);
        Y = yFold * pinv(hidden_outputs) * H;

        %% Storing the mean square error

        mse = sqrt( sum((Y - yTestFold) .^2) );
        mse_fold = mse_fold + mse;
    end

    mse_vector(k) = mse_fold / n_folds;
end

[min,centerQTD] = min(mse_vector);

%% Training the network

hidden_weights = rand(centerQTD, 1);
hidden_outputs = sigmf(hidden_weights * X_training, [1 0]);

%% Testing the network

H2test = sigmf(hidden_weights * X_test, [1 0]);
output = Y_training * pinv(hidden_outputs) * H2test;

%% Mean square error

mse = sqrt(sum((output - Y_test).^2));

%% Ploting results for the report

hold on
plot(X_test, output, 'r*')
plot(X_test, Y_test, 'b.')

%% Displaying summarized results

% fprintf('\nSummarizing, the radial basis function network was tested with...\n');
% fprintf('\tMean accuracy: %.2f\n', mean(accuracies));
% fprintf('\tMinimum accuracy: %.2f\n', min(accuracies));
% fprintf('\tMaximum accuracy: %.2f\n', max(accuracies));
% fprintf('\tStandard Deviation: %.2f\n\n', std(accuracies));

%% (un)Setting MATLAB Configurations

format;
