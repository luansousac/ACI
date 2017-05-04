function [ W ] = initWeights( n_rows, n_cols, a, b )
%INITWEIGHTSFOR Summary of this function goes here
%   Detailed explanation goes here
%   Making the data symmetric on line 11

    if nargin < 2
        n = 1;
    end

	bias = (b-a) * rand(n_rows, 1) + a;
    W = (b-a) * rand(n_rows, n_cols-1) + a;
    %W = rand(n, size(data, 2) - n) - 0.5;
    W = [bias W];

end
