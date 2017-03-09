function [T, W, N] = preprocess(data)
%% PREPROCESS
%    De uma forma geral, antes do treinamento para todos os algoritmos, a
%    partir do conjunto T (conjunto com todos os 310 padrões de indivíduos
%    normais e com patologias da coluna vertebral) deve-se criar três con-
%    juntos: um conjunto W sem amostras discrepantes, um conjunto N com as
%    amostras que estão em T e não estão em W, e um terceiro conjunto S que
%    relaciona o primeiro conjunto com o segundo da seguinte forma S = W U
%    P (N, p), em que P (N, p) representa um subconjunto de N dado por uma
%    percentagem p de amostras selecionadas aleatoriamente.
%
%    A forma de separação para se criar os conjuntos W e N segue os proce-
%    dimentos que são: remoção de padrões discrepantes e data clean. O pri-
%    meiro procedimento divide-se em duas tarefas. A primeira tarefa está
%    relacionada com a seleção de amostras, nas quais os valores para as
%    componentes estejam fora da faixa ?i ? 2?i > xi > ?i + 2?i, em que ?i
%    e ?i são a média e o desvio padrão da i-ésima componente da amostra x.
%    A segunda tarefa refere-se a exclusão dentre as amostras selecionadas
%    daquelas que apresentam uma ou mais componentes fora da faixa descrita.

    data_stdv = std(data);  % Desvio padrão de cada atributo
    data_mean = mean(data); % Média de cada atributo

    rows = size(data, 1);   % Quantidade de amostras
    cols = size(data, 2);   % Quantidade de atributos

    A = zeros(rows,cols);   % Matriz intermediária
    B = [];                 % Matriz intermediária

%% Remoção de dados discrepantes

    % Here we select logically only the elements that belong to the range:
    % [µj - 2*?j, µj + 2*?j], where µ is the mean of the jth attribute and
    % ? is the standard deviation of the jth attribute. So, we are gonna
    % have a matrix with 0s and 1s.
    for i = 1:cols
        A(1:rows, i) = data(1:rows, i) > (data_mean(i) - 2*data_stdv(i))...
                                          &...
                       data(1:rows, i) < (data_mean(i) + 2*data_stdv(i));
    end

    for i = 1:cols
        C = find(A(:,i) == 0);
        B = vertcat(B, C);
    end

    T = data;
    W = data;
    B = unique(B);
    N = data(B(:),:);
    W(B(:),:) = [];

% For the S set, we are gonna use the percents of N to do the union
% with W, bit a bit.

%% Data clean

% TO DO

end