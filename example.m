clc;
clear all;

% load dataset
dataset = 'delicious';
load (['dataset/', dataset, '.mat']);

% the number of labels
L = size(data.Y, 2);

% sort label acorrding to frequencies in ascending order
[~, pos] = sort(sum(data.Y, 1));
data.Y = data.Y(:, pos);
data.Yt = data.Yt(:, pos);

% try to load pre-trained model if it exists;
% otherwise it will train binary relevance model here 
filename = ['model/', dataset, '_model', '.mat'];

if exist(filename, 'file') == 2
    load (filename, 'W');
else
    q = '-s 0 -c 0.1 -B -1 -q';
    if strcmp(dataset, 'delicious') == 1
        q = '-s 3 -c 0.1 -B -1 -q';
    end
    [W] = BR_train(data.X, data.Y, q);
    save(filename, 'W');
end


M = zeros(size(data.X, 2), L);
for i = 1: L
    M(:, i) = W{i}.w;
end


tmp = whos('M');

% get the performance (pstart) of orginial model w.r.t P@k here.
tmpY = BR_predict(data.Xt, data.Yt, M);
[~, positions] = sort(tmpY, 2, 'descend');

% prev_topk: P@k of the original uncompressed model
pred = positions(:, 1:5);
prev_topk = topK(data.Yt, pred) * 100;

% average performance of P@k for k = 1 to 5
pstar = sum(prev_topk) / 3;

% store the original model size
prev_model_size = nnz(M);

% set performance tolerance level \epsilon to 1% here
% eps: hyperparameter to tune
eps = 0.1;
% perform label parameter opt.
[thre] = solver(data.Xt, data.Yt, M, pstar, eps);

% preserev top \delta largest absolute values for less-performance
% influential labels
% delta: hyperparameter to tune
delta = max(50, size(data.X, 2) * 0.05);
for i = 1: thre - 1
    [~, idx] = sort(abs(M(:, i)), 'descend');
    M(idx(delta + 1:end), i) = 0;
end

% feature parameter opt.
% alpha: hyperparameter to tune
alpha = 0.01;
M(abs(M) <= alpha) = 0;

% compute the performance of the final model
tmpY = BR_predict(data.Xt, data.Yt, M);
[~, positions] = sort(tmpY, 2, 'descend');
pred = positions(:, 1:5);

% topk: the final P@k
topk = topK(data.Yt, pred) * 100;
M = sparse(M);
% model_size: the final model size after pruning
model_size = nnz(M);
    

