function [ children ] = crossover_ox2( parents, fitness, population )

    n = size(parents, 1);
    n_cities = size(population, 2);
    children = zeros(n, n_cities);

    combination = randperm(n);
    n_offspring = floor(log2(n_cities));

    for c = 1 : 2 : n
        parent1_ind = combination(c);
        parent2_ind = combination(c+1);

        if fitness(parent1_ind) >= fitness(parent2_ind)
            winner = population(parent1_ind, :);
            loser  = population(parent2_ind, :);
        else
            winner = population(parent2_ind, :);
            loser  = population(parent1_ind, :);
        end

        sort_ind = sort(randperm(n_cities, n_offspring));
        sort_va1 = winner(sort_ind);
        sort_va2 = loser(sort_ind);

        idxs1 = arrayfun(@(x)find(loser == x, 1), sort_va1);
        idxs2 = arrayfun(@(x)find(winner == x, 1), sort_va2);

        child1 = winner;
        child2 = loser;
        child1(sort_ind) = loser(sort(idxs1));
        child2(idxs1) = sort_va1;

        children(c, :) = child1;
        children(c+1, :) = child2;
    end

end