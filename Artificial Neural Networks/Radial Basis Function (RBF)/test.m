function z = test(input, centers, betas, Theta)
    input(:,1) = [];%-1 * input(:,1);
    differences = bsxfun(@minus, centers, input);
    sqrdDifference = sum(differences .^ 2, 2);
    phis = [1; exp(-betas .* sqrdDifference)];
    z = Theta' * phis;
end