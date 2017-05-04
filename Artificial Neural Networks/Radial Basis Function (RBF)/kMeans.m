function [ centroids, memberships ] = kMeans( data, K )

    % Primeiro temos que selecionar os centroides iniciais. Tais centroides
    % serão escolhidos aleatoriamente para que os resultados finais não se
    % repitam.

    % Segundo: temos que separar os dados em K clusters.

    % Terceiro: temos que calcular as distâncias (euclidianas) mínimas de cada
    % objeto para cada um dos centroides.

    % Quarto: temos que checar se houve convergêcia dos clusters. Para isso
    % teremos variáveis de controle que guardarão os valores dos centroides
    % de cada iteração para fins de comparação.

    % Quinto: para cada centroide, se nenhum ponto foi atribuído ao centroide,
    % não mude-o. Caso contrário, compute o novo centroide do cluster selecio-
    % nando os dados atribuídos ao centroide 'k' e computando o novo centroide
    % como a média dos dados.

    n_patterns   = size(data, 1);
    n_attributes = size(data, 2);
    memberships  = zeros(n_patterns, 1);
    distances    = zeros(n_patterns, K);

    permutePatterns = randperm(n_patterns);
    previousCentroids = zeros(K, n_attributes);
    centroids = data(permutePatterns(1:K), :);

    for i = 1 : 100
        k = size(centroids, 1);

        for j = 1 : k
            difference = bsxfun(@minus, data, centroids(j, :));
            sqrdDifference = difference .^ 2;
            distances(:, j) = sum(sqrdDifference, 2);
        end

        [~, memberships] = min(distances, [], 2);

        for l = 1 : K
            if ~any(memberships == l)
                centroids(l, :) = centroids(l, :);
            else
                points = data((memberships == l), :);
                centroids(l, :) = mean(points);
            end
        end

        if previousCentroids == centroids
%             fprintf('Clusterização interrompida após %d iterações\n', i);
            break;
        end

        previousCentroids = centroids;
    end

end
