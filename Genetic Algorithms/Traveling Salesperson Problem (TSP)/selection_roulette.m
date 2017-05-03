function [ parents ] = selection_roulette( fitness, n_parents )

    wheel = cumsum(fitness) / n_parents;    % Vari�vel que armazena a soma das aptid�es
    parents = zeros(n_parents, 1);          % Matriz para armazenar os indiv�duos
    selected_fit = zeros(n_parents, 1);     % Matriz para armazenar as aptid�es

    for i = 1 : n_parents
        r = rand; % N�mero aleat�rio no intervalo (0,t)

        for j = 1 : length(wheel)
            if r < wheel(j)
                parents(i) = j;
                selected_fit(i) = j;
                break;
            end
        end
    end

end