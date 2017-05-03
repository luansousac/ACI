function [ parents ] = selection_roulette( fitness, n_parents )

    wheel = cumsum(fitness) / n_parents;    % Variável que armazena a soma das aptidões
    parents = zeros(n_parents, 1);          % Matriz para armazenar os indivíduos
    selected_fit = zeros(n_parents, 1);     % Matriz para armazenar as aptidões

    for i = 1 : n_parents
        r = rand; % Número aleatório no intervalo (0,t)

        for j = 1 : length(wheel)
            if r < wheel(j)
                parents(i) = j;
                selected_fit(i) = j;
                break;
            end
        end
    end

end