function [ W ] = init_weights( data, n )
%INITWEIGHTS Summary of this function goes here
%   Detailed explanation goes here
%   Making the data symmetric on line 11

    if nargin < 2
        n = 1;
    end

    bias = rand(n, 1);
    W = rand(n, size(data, 2) - n) - 0.5;
    W = [bias W];

end
