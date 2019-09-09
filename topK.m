function ratio = topK(Y, pred)


topK = [1, 3, 5];


ratio = zeros(length(topK), 1);

% if size(pred, 2) > 5
%     [~, positions] = sort(pred, 2, 'descend');
%     pred = positions(:, 1:5);
% end



for i = 1: length(topK)
    
    count = 0;
    for j = 1: topK(i)
        for k = 1: size(pred, 1)
            if pred(k, j) > 0 && Y(k, pred(k, j)) == 1
                count = count + 1;
            end
        end
    end
    
    ratio(i) = count / ( topK(i) * size(Y, 1) );
    
end