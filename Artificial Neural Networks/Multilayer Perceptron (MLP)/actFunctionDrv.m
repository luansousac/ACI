function [derivative] = actFunctionDrv(layer)

    derivative = exp(-layer) ./ ((1 + exp(-layer)) .* (1 + exp(-layer)));

end

