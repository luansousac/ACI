function plotarErros( vetordeErrosLineares )
%PLOTARERROS Summary of this function goes here
%   Detailed explanation goes here

    % Evolu��o do erro quadr�tico linear por �poca
    plot(vetordeErrosLineares, '^-')
    grid off
    xlabel('�poca')
    ylabel('Erro Quadr�tico M�dio')
    title('Erro Quadr�tico M�dio acumulado por �poca de Treinamento')

end
