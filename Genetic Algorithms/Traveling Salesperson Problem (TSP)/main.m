%% Setting useful MATLAB Configurations.

clc;
close all;
clearvars;
format long;

%% Displaying some info about the script.

fprintf('+----------------------------------------------------------------+\n');
fprintf('|                     INTELIGÊNCIA ARTIFICIAL                    |\n');
fprintf('|                  TRAVELING SALESPERSON PROBLEM                 |\n');
fprintf('|                 Prof.: Dr. Ajalmar R. Rocha Neto               |\n');
fprintf('|                  Acadêmico: Luan Sousa Cordeiro                |\n');
fprintf('|                              IFCE                              |\n');
fprintf('+----------------------------------------------------------------+\n');

disp(' ');
disp(' ');
disp('Press any key to continue...');
pause

%% Some important variables.

s_individual   = 40;   % Tamanho de cada indivíduo
n_individuals  = 60;   % Número de indivíduos por geração
n_generations  = 100;  % Número de gerações
t_crossover    = 0.8;  % Taxa de reprodução
t_mutation     = 0.1;  % Taxa de mutação
t_elitism      = 0.1;  % Taxa de elitismo

xindividuo = (1 : s_individual);
xfitness = (1 : n_generations);
vMediafitness = (1 : n_generations);
vMinfitness = (1 : n_generations);
vMaxfitness = (1 : n_generations);
xfuncaoObj = (1 : n_generations);

%% Generating random cities and its distances to each other.

n_cities = s_individual;               % Número de cidades para o PCV
cities = randn(n_cities, 2); % Cidades do Problema do Caixeiro Viajante
distances = zeros(n_cities); % Distâncias entre cada par de cidades

%% Importing the map borders from USA

load('usborder.mat','x','y');

%% Map type: Circle

% We will generate lineared spaced locations of cities so they can shape a circle.
% We use a desired radius and angles evenly spread around the circle, from 0 to 2*pi
% radians in order to create the matrix of city locations.

% radius = 1; 
% angles = linspace(0,2*pi,n_cities)';
% cities = radius*[sin(angles) cos(angles)];

%% Map type: Random

% We will generate random locations of cities inside the border of the
% United States. We can use the |inpolygon| function to make sure that all
% the cities are inside or very close to the US boundary.

n = 1;
while (n <= n_cities)
    xp = rand * 1.5;
    yp = rand;
    if inpolygon(xp, yp, x, y)
        cities(n,1) = xp;
        cities(n,2) = yp;
        n = n + 1;
    end
end

%% Calcuting the distances between the cities

for i = 1 : n_cities
    distances(i,:) = pdist2(cities(i,:), cities);
end

%% Creating a random initial population.

population = zeros(n_individuals, s_individual); % População inicial...
population_idxs = 1 : n_individuals;

for i = 1 : n_individuals
    population(i,:) = randperm(s_individual); % ... com genomas aleatórios
end

%% Genetic algorithm's loop. The algorithm stops when one of the stopping criteria is met.

for g = 1 : n_generations
    %% Computing the fitness value of each member from the current population.

    path_costs = compute_fitness(population, distances);

    %% Scales the raw fitness scores to convert them into a more usable range of values.

    fitness = fitscalingrank(path_costs, round(n_individuals * t_crossover));
    vMediafitness(g) = mean(fitness);
    vMinfitness(g) = min(fitness);
    vMaxfitness(g) = max(fitness);

    [~, idc] = min(fitness); % Valor e índice do fitness do melhor estado
    xindividuo(g,:) = population(idc,:); % Melhor individuo da geração
    xfitness(g)= fitness(idc); % Fitness do melhor indivíduo
    xfuncaoObj(g) = path_costs(idc);

    % Guarda o melhor estado da época anterior (elistimo)
    if (g > 1)
        if (xfitness(g) < xfitness(g-1))
            xindividuo(g,:) = xindividuo(g-1,:);
            xfitness(g) = xfitness(g-1);
            xfuncaoObj(g) = xfuncaoObj(g-1);
        end
    end

    %% Passing some individuals that have higher to the next population by elitism.

    [~, elit] = sort(fitness, 'descend');

    % Number of individuals to be kept as elit.
    n_elit = round(n_individuals * t_elitism);

    % Keeping the elit indexes.
    elit = elit(1:n_elit);

    % Setting the difference between initial and elit indexes.
    current_population = setdiff(population_idxs, elit);

    %% Selects members, called parents, based on their fitness (Roulette method).

    % Number of individuals to be selected as parents.
    n_parents = round(n_individuals * t_crossover);

    % Selecting the best individuals to be parents.
    parents = selection_roulette(fitness, n_parents);

    %% Produces children from the parents by order-based crossover (OX2).

     children = crossover_ox2(parents, fitness, population);

    %% Produces children from the parents by order-based mutation (OBM).

    mutate_idxs = setdiff(parents', current_population);
    n_mutation = length(mutate_idxs);

    for i = 1 : n_mutation
        pos1 = 0;
        pos2 = 0;

        while pos1 == pos2
            pos1 = round((n_cities-1) .* rand + 1);
            pos2 = round((n_cities-1) .* rand + 1);
        end

        aux = population(mutate_idxs(i), pos1);
        population(mutate_idxs(i), pos1) = population(mutate_idxs(i), pos2);
        population(mutate_idxs(i), pos2) = aux;
    end

    %% Replaces the current population with the children to form the next generation.

    next_population = [];
    next_population = [next_population; children];
    next_population = [next_population; population(elit, :)];
    next_population = [next_population; population(mutate_idxs, :)];
    population = next_population;
end

%% Displaying summarized results

fprintf('\n\nBusca em: 100'); fprintf(' por cento\n\n');
[val, idc] = sort(xfuncaoObj);
fprintf('\nGerando os resultados e os gráficos... \n');
fprintf('-------------------------------\n\n');

fprintf('* Geração: %d\n', g);
fprintf('* Melhor Estado:\n\n')
disp(xindividuo(g-1,:));
fprintf('* Função de Fitness = %f\n', xfitness(g-1));
fprintf('* Função Objetivo = %f\n', xfuncaoObj(g-1));

%% Plotting the USA map and the random cities

figure
hold on
axis equal
plot(x, y, 'Color', 'red')
plot_solution(cities, xindividuo(g-1,:));
hold off

%% Gráficos: Fitness e Função Objetivo

figure
subplot(2, 1, 1)
plot(xfitness, 'k')
legend('Fitness')
xlabel('Geração')

subplot(2, 1 ,2)
plot(xfuncaoObj, 'b')
legend('F. Objetivo')
xlabel('Geração')

figure
hold on
plot(vMediafitness, 'r', 'DisplayName', 'Média')
plot(vMinfitness, 'b', 'DisplayName', 'Mínimo')
plot(vMaxfitness, 'g', 'DisplayName', 'Máximo')
legend('Location', 'best')
legend show
xlabel('Geração')
ylabel('Função de aptidão')

%% (un)Setting MATLAB Configurations

format;
