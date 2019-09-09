function [W] = BR_train(X_train, Y_train, p)

%function [Y_predict, dec_val, W, test_time] = BR(X_train, Y_train, X_test, Y_test, p,  delta)
% binary relevance method
% X_train: n1 by m matrix, each line is a feature vector of an instance
% Y_train: n1 by k matrix
% X_test : n2 by m matrix
% Y_test : n2 by k matrix
% Y_predict: n2 by k matrix

    %addpath('../libsvm/matlab');
    addpath('../liblinear/matlab');
    
    nLabels = size(Y_train, 2);
    %nTest   = size(X_test, 1);

    W = cell(nLabels, 1);
    X_train = sparse(X_train);
    
    parfor L = 1: nLabels
        model = train(full(Y_train(:, L)), X_train, p);
        
        if model.Label == [0;1]
            model.Label = [1;0];
            model.w = -model.w;
        end
        
        %model.w(abs(model.w) <= delta) = 0;
        %model.w = sparse(model.w);
        W{L} = model;
    end
    
%     X_test = sparse(X_test);
%     tic
%     for L = 1: nLabels
%         model = W{L};
%         model.w = full(model.w);
%         [~, ~, dec] = predict(full(Y_test(:, L)), X_test, model,  '-q');
%         
%         idx = find(dec_val(:, 5) < dec);
%         Y_predict(idx, 5) = L;
%         dec_val(idx, 5) = dec(idx);
%         
%         
%        [dec_val, positions] = sort(dec_val, 2, 'descend');
%        for i = 1: nTest
%            Y_predict(i, :) = Y_predict(i, positions(i, :));
%        end
%     end
%     test_time = toc;

end
