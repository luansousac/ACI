function [ data ] = shuffle_data( data )
%SHUFFLE_DATA shuffles the presented data set in order to achieve different
% results at each execution. This is a general function, working for any
% data set. It works as follows:
%       1. Generates random indices (integers) from 1 to number of patterns
%		   (samples, rows) from the data set.
%       2. Uses these random indices to shuffle the original data set.
%
% Example:
%    shuffled_data = shuffle_data( data );
%
% This function also works in conjunction with SPLIT_DATA.

    n_patterns = length(data);
    index = randperm(n_patterns);
    data = data(index, :);
end
