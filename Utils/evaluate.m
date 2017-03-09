function [ Y ] = evaluate( Y )
%EVALUATE evaluates a given output that is in the interval [a,b].
%   EVALUATE function works for problems with two or more classes.
%   If it's a problem with 2 (two) classes, just apply the step function.
%   If it's a problem with more than 2 (two) classes, select the maximum
%   values from each column of the matrix 'Y' and set them to 1, with the
%   rest of the rows of such column being set to 0.

    [x,y] = size(Y);

    if x == 1
        Y = (Y >= 0);
    else
        [~, i] = max(Y);
        Y = zeros(x, y);
        for j = 1 : y
            Y(i(j), j) = 1;
        end
    end

end
