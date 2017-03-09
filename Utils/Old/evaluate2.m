function [ Y ] = evaluate2( Y, k )
%RE Summary of this function goes here
%   Detailed explanation goes here

    if k == 1
        [s, i] = sort(Y, 'descend');
        r = randperm(2);
        Y(i(r(1))) = 1;
        Y(i(r(2))) = 0;
        a = (Y == s(3));
        Y(a) = 0;
    elseif k == 2
        [~, i] = max(Y);
        a = [1 2 3];
        a(i) = [];
        Y(i) = 1;
        Y(a) = 0;
    end

end

