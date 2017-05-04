function plotarErros( vetordeErrosLineares )
%PLOTARERROS Summary of this function goes here
%   Detailed explanation goes here

    % Evolução do erro quadrático linear por época
    plot(vetordeErrosLineares, '^-')
    grid off
    xlabel('Época')
    ylabel('Erro Quadrático Médio')
    title('Erro Quadrático Médio acumulado por Época de Treinamento')

end
