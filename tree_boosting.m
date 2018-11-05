% Based on code from:
% https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d

x = [1:50]';
y = [rand(1,10),rand(1,10)+4, rand(1,10)-2, rand(1,10)+1, rand(1,10)+3]';
error = y;

% Initial prediction is just 0
prediction = 0;
error = y;

for i = 1:20
    T = DecisionTree(x, error, 1:length(error), 1, 1);
    pred = DecisionTree_prediction(T, x);
    prediction = prediction + pred;
    
    norm(pred)
    
    prev_error = error;
    error = y - prediction;
    norm(error - prev_error)
    
    figure()
    plot(x, y, 'rx')
    hold on
    plot(x, prediction)
    title(['iteration:', num2str(i)])
end

    

function T = DecisionTree(x, y, index, min_leaf, max_depth)
% Construct a single decision tree on the data (x, y), where each leaf has
% a minimum of min_leaf datapoints associated to that node.

    % Store all information about tree in structure
    T.x = x;
    T.y = y;
    T.index = index;
    T.min_leaf = min_leaf;
    T.max_depth = max_depth;
    T.val = mean(y(index)); % Value of node
    T.score = inf; % Tree scoring, a score of inf means leaf node
    T.depth = 0;
    
    % Run through columns of data trying to find 'best' split.
    for i = 1:size(x,2)
         T = find_split(T, i);
    end
    
    index = 1:length(T.index); % used to prevent 'find'
    if T.score ~= inf
        T.depth = T.depth +1;
        
        % Select column of x, which we are using to split the data
        x = T.x(T.index, T.col_index);
        
        % Recursively construct two new trees, one for each branch of the
        % split. Give the index of the x and y in that branch.
        T.lhs = DecisionTree(T.x, T.y, T.index(index(x<=T.split)), min_leaf, T.max_depth - T.depth);
        T.rhs = DecisionTree(T.x, T.y, T.index(index(x>T.split)), min_leaf, T.max_depth - T.depth);
    end
end

function T = find_split(T, col_index)
% Search for 'best' split at current node

    % Select the training data at this current node and sort.
    x = T.x(T.index, col_index);
    y = T.y(T.index);
    [sort_x, sort_index] = sort(x);
    sort_y = y(sort_index);
    
    % construct lhs and rhs pivots to find split.
    rhs = [length(T.index), sum(sort_y), sum(sort_y.*sort_y)];
    lhs = [0, 0, 0];
    
    % Each branch must contain min_leaf datapoints so dont need to run
    % through to end.
    for i = 1:(length(T.index) - T.min_leaf)
        xi = sort_x(i);
        yi = sort_y(i);
        lhs = lhs + [1, yi, yi^2];
        rhs = rhs - [1, yi, yi^2];

        % Ensure lhs has min_leaf and datapoints of same value are in same
        % branch
        if i < T.min_leaf || xi == sort_x(i+1) || T.depth == T.max_depth
            continue
        end
        
        lhs_std = sqrt(lhs(3)/lhs(1) - (lhs(2)/lhs(1))^2);
        rhs_std = sqrt(rhs(3)/rhs(1) - (rhs(2)/rhs(1))^2);
        current_score = lhs_std*lhs(1) + rhs_std*rhs(1);
        
        if current_score < T.score
            % Column of current optimal split
            T.col_index = col_index;
            T.score = current_score;
            % Value of where split on this column
            T.split = xi;
            
        end
    end
end

function pred = DecisionTree_prediction(T, x)
% Run through the rows of x and make a prediction for each datapoint
    pred = zeros(size(x,1),1);
    for i =  1:size(x,1)
        pred(i) = predict_recursion(T, x(i,:));
    end
end

function pred = predict_recursion(T, xi)
% Recurse down the tree, once reached leaf node, predict values
    if T.score == inf
       pred = T.val;
    else
        % Test value vs split value and choose which branch
        if xi(T.col_index)<=T.split
            subtree = T.lhs;
        else
            subtree = T.rhs;
        end
        pred = predict_recursion(subtree, xi);
    end
end