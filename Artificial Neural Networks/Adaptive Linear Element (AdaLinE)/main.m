%% Setting useful MATLAB Configurations

clc;
close all;
clearvars;
format LONG;

%% Adding important paths into the workspace

addpath('../../Datasets/airfoil_self_noise/');

%% Diplaying the beginning of the script

fprintf('+----------------------------------------------------------------+\n');
fprintf('|               INTELIG�NCIA COMPUTACIONAL APLICADA              |\n');
fprintf('|                 REDE ADA(ptive)LIN(ear)E(lement)               |\n');
fprintf('|                 Prof.: Dr. Ajalmar R. Rocha Neto               |\n');
fprintf('|                  Acad�mico: Luan Sousa Cordeiro                |\n');
fprintf('|                              IFCE                              |\n');
fprintf('+----------------------------------------------------------------+\n');

disp(' ');
disp(' ');

%%

disp('Configura��es:')
disp(' ')
NumeroPadroes = input('N�mero de amostras: ');
Desvio = input('Desvio padr�o para o ru�do: ');
a = input('Inclina��o da reta: ');
b = input('Termo Independente da reta: ');
Eta = input('Taxa de aprendizado: ');
MaxEpocas = input('N�mero m�ximo de �pocas de treinamento: ');
Tolerancia = input('Toler�ncia para a diferen�a de erros: ');
disp(' ')

%%

%--------------------------------------------------------------------------
% Entradas e sa�das desejadas

% Entradas
X = linspace(-2,2,NumeroPadroes)';

% Sa�das desejadas - reta com ru�do
Ruido = Desvio*randn(NumeroPadroes,1);
Yd = a*X + b + Ruido;
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Inicializa��es

% Dimens�o dos padr�es de entrada
N = size(X,2); 

% Adicionando a entrada de bias
X_Bias = 1*ones(NumeroPadroes, 1);
X = [X X_Bias];

% Inicializando vetor de pesos (o �ltimo peso � o bias) aleatoriamente, com
% distribui��o normal de m�dia zero e desvio padr�o 1
W     = randn(1, N); 
Bias  = randn();

% Adicionando o bias � matriz de pesos
W = [W Bias]; 

Erro = 0;

Vetor_Erros = [];

Epoca = 1;
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Loop de treinamento

while (true)  

    % Erro quadr�tico para a �poca
    Erroq = 0;  

	% Para treinar com um padr�o aleat�rio, cria-se um vetor com �ndices
    % aleat�rios
    Idx = randperm(NumeroPadroes);
	
    %**********************************************************************
    % Cada ciclo completo do for representa uma �poca de treinamento: 
    % em cada �poca, todos os padr�es de treinamento s�o apresentados,
    % aleatoriamente, � rede
    
    for i = 1 : NumeroPadroes
        
        % Calcula sa�da
	    y = W*X(Idx(i), :)';
        
        % Calcula erro 
        Erro = (Yd(Idx(i)) - y)';
      
        % Calcula o vetor Delta_W
        Delta_W = Eta*Erro*X(Idx(i), :);
        
        % Atualiza vetor de pesos
        W = W + Delta_W;
        
        % Atualiza o erro quadr�tico da �poca
        Erroq = Erroq + Erro^2;
        
    end % for i = 1 : NumeroPadroes
    %**********************************************************************
	
	% Armazena erro quadr�tico acumulado para cada �poca
    Vetor_Erros = [Vetor_Erros; Erroq];

	%**********************************************************************
    % Visualiza��o gr�fica da evolu��o do treinamento.
    clf
    plot(X(:, 1), Yd,'r.')
	axis ([-2 2 min(Yd)-1 max(Yd)+1])
    grid on
    hold on
    
    % Reta aproximada pelo Adaline
    Reta = W(1)*X + W(2);
    plot(X,Reta,'k')
    title('Evolu��o da Aproxima��o Obtida')
    xlabel('x')
    ylabel('f(x)')
    
    legend('Amostras', 'Reta Obtida', 'Location', 'NorthWest')
	
	% Pausa para melhor vizualiza��o da evolu��o da reta (aproxima��o)
    pause(0.25)
    %**********************************************************************
	
	if (Epoca > 1)
	
		% Encerra treinamento se o crit�rio de erro foi alcan�ado
		Diferenca = abs(Vetor_Erros(Epoca - 1) - Erroq);
		
		if ((Diferenca < Tolerancia) || (Epoca > MaxEpocas))
        
			break;
        
		end % if ((Diferenca < Tolerancia) || (Epoca > MaxEpocas))
		
	end % if (Epoca > 1)

	Epoca = Epoca + 1;

end % while (true)      
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Evolu��o do erro quadr�tico acumulado por �poca

figure
plot(Vetor_Erros, '^-')
grid on
xlabel('�poca')
ylabel('Erro Quadr�tico')
title('Erro Quadr�tico M�dio por �poca de Treinamento')
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Resultados

disp('__________________________________________________________________ ')
disp(' Resultados:')
disp(' ')
fprintf('   >>> O valor obtido para o peso 1 foi: %f\n', W(1))
fprintf('   >>> O valor obtido para o bias foi: %f\n', W(2))
disp('__________________________________________________________________ ')
disp(' ')
%--------------------------------------------------------------------------