function [ data ] = labelClasses( data, classes, opt )
%ROTULECLASSES Summary of this function goes here
%   Detailed explanation goes here

% Aqui introduzimos vetores bin�rios para representar cada classe. Assim
% classes(i,1) representa o diagn�stico do paciente i, enquanto classes(i,2)
% representa sua codifica��o em vetor bin�rio.

% Bom ver se d� pra usar o cellfun combinado com o de2bi

	n_features = size(data, 2);

	switch opt
		case 1
			class1 = find(data(:,n_features) == 1);
			class2 = find(data(:,n_features) == 2);
		case 2
			class1 = find(strcmp(classes, 'Iris-setosa'));
			class2 = find(strcmp(classes, 'Iris-versicolor'));
			class3 = find(strcmp(classes, 'Iris-virginica'));
		case 3
			class1 = find(strcmp(classes, 'DH'));
			class2 = find(strcmp(classes, 'SL'));
			class3 = find(strcmp(classes, 'NO'));
	end

	if opt ~= 1
		data(class1, n_features + 1) = 1;
		data(class1, n_features + 2) = 0;
		data(class1, n_features + 3) = 0;
		data(class2, n_features + 1) = 0;
		data(class2, n_features + 2) = 1;
		data(class2, n_features + 3) = 0;
		data(class3, n_features + 1) = 0;
		data(class3, n_features + 2) = 0;
		data(class3, n_features + 3) = 1;
	end
end

