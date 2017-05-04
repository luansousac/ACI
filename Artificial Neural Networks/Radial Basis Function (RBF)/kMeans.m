function [ centroids, memberships ] = kMeans( data, K )

    % Primeiro temos que selecionar os centroides iniciais. Tais centroides
    % ser�o escolhidos aleatoriamente para que os resultados finais n�o se
    % repitam.

    % Segundo: temos que separar os dados em K clusters.

    % Terceiro: temos que calcular as dist�ncias (euclidianas) m�nimas de cada
    % objeto para cada um dos centroides.

    % Quarto: temos que checar se houve converg�cia dos clusters. Para isso
    % teremos vari�veis de controle que guardar�o os valores dos centroides
    % de cada itera��o para fins de compara��o.

    % Quinto: para cada centroide, se nenhum ponto foi atribu�do ao centroide,
    % n�o mude-o. Caso contr�rio, compute o novo centroide do cluster selecio-
    % nando os dados atribu�dos ao centroide 'k' e computando o novo centroide
    % como a m�dia dos dados.

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
%             fprintf('Clusteriza��o interrompida ap�s %d itera��es\n', i);
            break;
        end

        previousCentroids = centroids;
    end

end
