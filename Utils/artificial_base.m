%% Example of an artificial data set in 2-D with labels using the OvA approach.
%  Produced-by: Prof. Dr. Ajalmar R. Rocha Neto
%  Edited-by: Luan Sousa Cordeiro

% Cleaning the Command Window. Removing all variables. Closing all figures.
clc;
clearvars;
close all;

% Number of samples for each class.
N = 100;

% Standard deviation of the data.
st_d = 0.05;

% Region of space that the data in which the data is going to surround.
region1 = [0.3 0.3];
region2 = [0.5 0.8];
region3 = [0.8 0.5];

% Data from each class based on the standard deviation.
data_class1 = repmat(region1, N, 1) + st_d * randn(N,2);
data_class2 = repmat(region2, N, 1) + st_d * randn(N,2);
data_class3 = repmat(region3, N, 1) + st_d * randn(N,2);

% Grouped data into one big data set, giving them its lables.
data = [data_class1 repmat([1 0 0], N, 1);
        data_class2 repmat([0 1 0], N, 1);
        data_class3 repmat([0 0 1], N, 1)];
    
% Plotting the data set.
figure
hold on
axis([0 1 0 1])
title('2-D non-linearly separable data set')
plot(data_class1(:,1), data_class1(:,2), '*', 'Color', [0.60 0.20 0.00], ...
                                              'DisplayName', 'Class 1')
plot(data_class2(:,1), data_class2(:,2), '*', 'Color', [0.00 0.45 0.74], ...
                                              'DisplayName', 'Class 2')
plot(data_class3(:,1), data_class3(:,2), '*', 'Color', [0.31 0.31 0.31], ...
                                              'DisplayName', 'Class 3')
hold off
legend show

% Simulating a weight vector found in a perceptron training, for instance.
W = [1.0 2.0 -2.5];

% Loop for mapping the data in two decision regions
i = 1;
j = 1;
for x1 = 0:0.005:1
    for x2 = 0:0.005:1
        p = [x1 x2];
        u = [1 x1 x2] * W';
        if u >= 0
            C1(i,:) = p;
            i = i + 1;
        else
            C2(j,:) = p;
            j = j + 1;
        end
    end
end

% Plotting the map color with the boundary divisions
figure
hold on
axis([0 1 0 1])
title('Map color for boundary decision in 2-D')
plot(C1(:,1), C1(:,2), '.', 'Color', [0.00 0.45 0.74], 'LineStyle', '-', ...
                            'LineWidth', 3.0, 'HandleVisibility', 'off')
plot(C2(:,1), C2(:,2), '.', 'Color', [0.80 0.88 0.97], 'LineStyle', '-', ...
                            'LineWidth', 3.0, 'HandleVisibility', 'off')
hold off
