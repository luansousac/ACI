%% Setting useful MATLAB Configurations

clc;
close all;
clearvars;
format LONG;

%% Adding important paths into the workspace

addpath('../../Datasets/airfoil_self_noise/');

%% Diplaying the beginning of the script

fprintf('+----------------------------------------------------------------+\n');
fprintf('|               INTELIGÊNCIA COMPUTACIONAL APLICADA              |\n');
fprintf('|                 REDE ADA(ptive)LIN(ear)E(lement)               |\n');
fprintf('|                 Prof.: Dr. Ajalmar R. Rocha Neto               |\n');
fprintf('|                  Acadêmico: Luan Sousa Cordeiro                |\n');
fprintf('|                              IFCE                              |\n');
fprintf('+----------------------------------------------------------------+\n');

disp(' ');
disp(' ');

%%

disp('Configurações:')
disp(' ')
NumeroPadroes = input('Número de amostras: ');
Desvio = input('Desvio padrão para o ruído: ');
a = input('Inclinação da reta: ');
b = input('Termo Independente da reta: ');
Eta = input('Taxa de aprendizado: ');
MaxEpocas = input('Número máximo de épocas de treinamento: ');
Tolerancia = input('Tolerância para a diferença de erros: ');
disp(' ')

%%

%--------------------------------------------------------------------------
% Entradas e saídas desejadas

% Entradas
X = linspace(-2,2,NumeroPadroes)';

% Saídas desejadas - reta com ruído
Ruido = Desvio*randn(NumeroPadroes,1);
Yd = a*X + b + Ruido;
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Inicializações

% Dimensão dos padrões de entrada
N = size(X,2); 

% Adicionando a entrada de bias
X_Bias = 1*ones(NumeroPadroes, 1);
X = [X X_Bias];

% Inicializando vetor de pesos (o último peso é o bias) aleatoriamente, com
% distribuição normal de média zero e desvio padrão 1
W     = randn(1, N); 
Bias  = randn();

% Adicionando o bias à matriz de pesos
W = [W Bias]; 

Erro = 0;

Vetor_Erros = [];

Epoca = 1;
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Loop de treinamento

while (true)  

    % Erro quadrático para a época
    Erroq = 0;  

	% Para treinar com um padrão aleatório, cria-se um vetor com índices
    % aleatórios
    Idx = randperm(NumeroPadroes);
	
    %**********************************************************************
    % Cada ciclo completo do for representa uma época de treinamento: 
    % em cada época, todos os padrões de treinamento são apresentados,
    % aleatoriamente, à rede
    
    for i = 1 : NumeroPadroes
        
        % Calcula saída
	    y = W*X(Idx(i), :)';
        
        % Calcula erro 
        Erro = (Yd(Idx(i)) - y)';
      
        % Calcula o vetor Delta_W
        Delta_W = Eta*Erro*X(Idx(i), :);
        
        % Atualiza vetor de pesos
        W = W + Delta_W;
        
        % Atualiza o erro quadrático da época
        Erroq = Erroq + Erro^2;
        
    end % for i = 1 : NumeroPadroes
    %**********************************************************************
	
	% Armazena erro quadrático acumulado para cada época
    Vetor_Erros = [Vetor_Erros; Erroq];

	%**********************************************************************
    % Visualização gráfica da evolução do treinamento.
    clf
    plot(X(:, 1), Yd,'r.')
	axis ([-2 2 min(Yd)-1 max(Yd)+1])
    grid on
    hold on
    
    % Reta aproximada pelo Adaline
    Reta = W(1)*X + W(2);
    plot(X,Reta,'k')
    title('Evolução da Aproximação Obtida')
    xlabel('x')
    ylabel('f(x)')
    
    legend('Amostras', 'Reta Obtida', 'Location', 'NorthWest')
	
	% Pausa para melhor vizualização da evolução da reta (aproximação)
    pause(0.25)
    %**********************************************************************
	
	if (Epoca > 1)
	
		% Encerra treinamento se o critério de erro foi alcançado
		Diferenca = abs(Vetor_Erros(Epoca - 1) - Erroq);
		
		if ((Diferenca < Tolerancia) || (Epoca > MaxEpocas))
        
			break;
        
		end % if ((Diferenca < Tolerancia) || (Epoca > MaxEpocas))
		
	end % if (Epoca > 1)

	Epoca = Epoca + 1;

end % while (true)      
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Evolução do erro quadrático acumulado por época

figure
plot(Vetor_Erros, '^-')
grid on
xlabel('Época')
ylabel('Erro Quadrático')
title('Erro Quadrático Médio por Época de Treinamento')
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