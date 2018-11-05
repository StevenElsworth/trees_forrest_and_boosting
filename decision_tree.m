% Based on code from:
% https://github.com/fastai/fastai/blob/master/courses/ml1/lesson3-rf_foundations.ipynb

x = [-1,3;-6,1;-3,4;1,1;2,5;3,1;1,-2;3,-2;2,-1;-2,-1;-1,-2;-1,-3];
y = [1,1,1,0,0,0,1,1,1,0,0,0];
index = 1:length(y);

T = DecisionTree(x, y, index, 2, 2);
DecisionTree_prediction(T, x);

plot_example_decision_tree(T)


function plot_example_decision_tree(T)
    figure()
    ind = find(T.y == 0);
    plot(T.x(ind, 1), T.x(ind, 2), 'rx')
    hold on 
    ind = find(T.y == 1);
    plot(T.x(ind, 1), T.x(ind, 2), 'bo')
    axis([-7,7,-7,7])

    recursive_plot(T);
    
    
    function recursive_plot(T)
        A = repmat(linspace(-7, 7, 10)', 1,2);
        A(:,T.col_index) = T.split*ones(10,1)+0.5;
        plot(A(:,1), A(:,2), 'k')
        
        if isfield(T, 'lhs')
            if isfield(T.lhs, 'split')
                recursive_plot(T.lhs)
            end
        end
        
        if isfield(T, 'rhs')
            if isfield(T.rhs, 'split')
                recursive_plot(T.rhs)
            end
        end    
    end
end

%F = RandomForrest(x, y, 40, 2, 3, 0.8);
%round(RandomForrest_prediction(F, x));



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
        if i < T.min_leaf || xi == sort_x(i+1)
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


function F = RandomForrest(x, y, number_of_trees, min_leaf, max_depth, sample_proportion)
% Construct a random forrest made up of 'number_of_trees' trees where each
% tree recieves 'sample_proportion' of the data
    F.x = x;
    F.y = y;
    F.sample_proportion = sample_proportion;
    F.number_of_trees = number_of_trees;
    F.max_depth = max_depth;
    
    for i = 1:number_of_trees
        % Create random permutation and select appropriate number of data
        % points
        index = randperm(length(y));
        index = index(1:round(sample_proportion*length(y)));
        % Construct decision tree on data sample.
        F.trees{i} = DecisionTree(x, y, index, min_leaf, max_depth);
    end

end

function val = RandomForrest_prediction(F, x)
% Recurse down each tree in the random forrest and take mean of the
% predictions.
    val = zeros(size(x, 1),1);
    for i = 1:F.number_of_trees
        % Predict for each decision tree
        val = val + DecisionTree_prediction(F.trees{i}, x);
    end
    % Take mean of predictions
    val = val./F.number_of_trees;
end
