function [lb] = solver(Xtest, Ytest, M, pstar, eps)
% Parameters:
% M: the uncompressed model
% pstar: the original performance
% eps: performance degenration tolerance
%
% Return:
% lb: the number of tail labels to remove

    Y = BR_predict(Xtest, Ytest, M);
    
    lb = 1;
    ub = size(M, 2) - 1;
    L = size(Y, 2);
    
    while ub - lb > 1
        mid = floor((lb + ub) / 2);
        tmpY = Y(:, mid:L);

        [~, positions] = sort(tmpY, 2, 'descend');
        pred = positions(:, 1:5);

        if sum(topK(Ytest(:, mid:L), pred)) * 100 / 3 >= pstar - eps
            lb = mid;
        else
            ub = mid;
        end
    end
end