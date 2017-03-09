function [ data ] = labelClasses( data, classes, opt )
%LABELCLASSES gives classes a binary label
%   This function cover the following data sets:
%		- Banknote Authentication, where the labels come from the data set
%		- Iris Flower, where setosa flowers are 1 and non-setosa flowers are 0
%		- Vertebral Column, where normal patients are 1 and disabled ones are 0

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

	data(class1, n_features + 1) = 1;
	data(class2, n_features + 1) = 0;
	if opt ~= 1
		data(class3, n_features + 1) = 0;
	end

end

