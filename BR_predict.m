function [dec_val] = BR_predict(X, Y_test, W)

dec_val = X * W;

% X_test = sparse(X);
% nLabels = size(W, 1);
% nTest = size(X, 1);
% 
% addpath('../liblinear/matlab');

%Y_predict = zeros(size(X, 1), nLabels);
%dec_val = zeros(nTest, nLabels);


% parfor L = 1: nLabels
%     model = W{L};
%     model.w = full(model.w);
%     [~, ~, dec] = predict(full(Y_test(:, L)), X_test, model,  '-q');
%     
%     %idx = find(dec_val(:, 5) < dec);
%     %Y_predict(idx, 5) = L;
%     %dec_val(idx, 5) = dec(idx);
%     dec_val(:, L) = dec;
%     
%     %[dec_val, positions] = sort(dec_val, 2, 'descend');
%     %for i = 1: nTest
%     %     Y_predict(i, :) = Y_predict(i, positions(i, :));
%     %end
% end
