function [ path_costs ] = compute_fitness( population, distances )

    n = size(population, 1);
    n_cities = size(population, 2);
    path_costs = zeros(n, 1); % Funções de aptidão

    % Soma das distâncias que formam o caminho dado pelo indivíduo
    for i = 1 : n
        S = 0;

        for j = 1 : n_cities
            if j == n_cities
                a = 1;
                b = population(i,j);
            else
                a = population(i,j);
                b = population(i,j+1);
            end

            S = S + distances(a,b);
        end

        path_costs(i) = S;
    end

end

