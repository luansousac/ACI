function [ centers, betas, Theta ] = train( data, dOutput, centersPerCategory, sigma )

    % ================================================= %
    %  0. Configurações necessárias para o treinamento  %
    % ================================================= %

    data(:,1) = [];%-1 * data(:,1);
    n_patterns   = size(data, 1);
    n_categories = size(unique(dOutput), 1);

    if any(dOutput == 0) || any(dOutput > n_categories)
        error('Os valores de categorias devem sem não-negativos e contínuos.');
    end

    % ================================================= %
    %    1. Selecionando os centros e os parâmetros     %
    % ================================================= %

    % disp('TESTANDO');
    % disp('1. Selecionando os centros através do k-Médias.');

    for c = 1 : n_categories
        % ================================================ %
        %     1.1. Encontrando os centros dos clusters     %
        % ================================================ %

        % fprintf('\t Calculando os centros da categoria %d...\n', c);

        dataForCategory = data(dOutput == c, :);
        [centroids, memberships] = kMeans(dataForCategory, centersPerCategory);

        % ================================================ %
        %         1.2. Removendo os clusters vazios        %
        % ================================================ %

        toRemove = [];
        for i = 1 : size(centroids, 1)
            if sum(memberships == i) == 0
                toRemove = [toRemove; i];
            end
        end

        if ~isempty(toRemove)
            centroids(toRemove, :) = [];
            k = size(centroids, 1);
            m = size(dataForCategory, 1);
            distances = zeros(m, k);

            % Para cada centroide, são calculadas as distâncias euclidianas.
            for i = 1 : k
                difference = bsxfun(@minus, dataForCategory, centroids(i,:));
                distances(:, i) = sum(difference .^ 2, 2);
            end

            [~, memberships] = min(distances, [], 2);
        end

        % ================================================ %
        %      2. Calculando os coeficientes Beta          %
        % ================================================ %

        % fprintf('\t Calculando os coeficientes beta da categoria %d...\n\n', c);

        n_hiddenNeuros = size(centroids, 1);
        sigmas = ones(n_hiddenNeuros, 1);

        for i = 1 : n_hiddenNeuros
            center = centroids(i, :);
            members = data(memberships == i, :);
            differences = bsxfun(@minus, members, center);
            sqrdDifferences = sum(differences .^ 2, 2);
            distances = sqrt(sqrdDifferences);
            %distances = sqrdDifferences;
            %sigmas(i, :) = mean(distances);
            sigmas(i, :) = sigma;
        end

        if any(sigmas == 0)
            error('Algum sigma tem valor zero! (Não pode acontecer)');
        end

        betas = sigma;%1 ./ (2 .* (sigmas .^ 2));
        centers = centroids;
    end

    % ================================================= %
    %      3. Treinando os pesos da camada de saída     %
    % ================================================= %

    % disp('2. Calculando as ativações dos neurônios RBF.');

    n_hiddenNeuros = size(centers, 1);
    X_activ = zeros(n_patterns, n_hiddenNeuros);

    for i = 1 : n_patterns
        input = data(i, :);

        % Get the activation for all RBF neurons for this input.
        differences = bsxfun(@minus, centers, input);
        sqrdDifference = sum(differences .^ 2, 2);
        z = exp(-betas .* sqrdDifference);

        % Store the activation values 'z' for training example 'i'.
        X_activ(i, :) = z';
    end

    % Add a column of 1s for the bias term.
    X_activ = [ones(n_patterns, 1) X_activ];

    % =============================================
    %        Learn Output Weights
    % =============================================

    % disp('3. Aprendendo os pesos de saída.');

    % Create a matrix to hold all of the output weights.
    % There is one column per category / output neuron.
    Theta = zeros(n_hiddenNeuros + 1, n_categories);

    % For each category...
    for c = 1 : n_categories
        % Make the y values binary: 1 for category 'c' and 0 for all other
        % categories.
        y_c = (dOutput == c);

        % Use the normal equations to solve for optimal theta.
        Theta(:, c) = pinv(X_activ' * X_activ) * X_activ' * y_c;
    end
    
end
