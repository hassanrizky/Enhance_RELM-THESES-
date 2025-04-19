% Load data
X = data(:, 2:101);
Y = data(:, 1);

% Set random seed
rng(82)
index00 = randperm(1000);

% Split data into training, validation, and testing sets
trainIndex = index00(1:600);
validationIndex = index00(601:700);
testIndex = index00(701:1000);

X_train = X(trainIndex, :);
Y_train = Y(trainIndex, 1);

X_val = X(validationIndex, :);
Y_val = Y(validationIndex, 1);

X_test = X(testIndex, :);
Y_test = Y(testIndex, 1);

% Train k-NN Regression model
k = 5; % Number of neighbors
tic;
modelKNN = fitcknn(X_train, Y_train, 'NumNeighbors', k, 'Distance', 'euclidean', 'Standardize', 1);
trainingTime = toc;

% Validate the model
tic;
Y_ValTest = predict(modelKNN, X_val);
validationTime = toc;

rmseVal = rmse(Y_ValTest', Y_val');
maeVal = mae(Y_ValTest', Y_val');
mapeVal = mape(Y_ValTest', Y_val');
rsquareVal = rsquare(Y_ValTest, Y_val);

% Test the model
tic;
Y_OutTest = predict(modelKNN, X_test);
executionTime = toc;

rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

% Display results
disp('---------k-Nearest Neighbors Regression----------');
disp('Validation Data');
disp(['RMSE Val KNN: ', num2str(rmseVal)]);
disp(['MAE Val KNN: ', num2str(maeVal)]);
disp(['MAPE Val KNN: ', num2str(mapeVal)]);
disp(['R-Square Val KNN: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test KNN: ', num2str(rmseTest)]);
disp(['MAE Test KNN: ', num2str(maeTest)]);
disp(['MAPE Test KNN: ', num2str(mapeTest)]);
disp(['R-Square Test KNN: ', num2str(rsquareTest)]);
disp(' ');
disp('Execution Time');
disp(['Training Time KNN: ', num2str(trainingTime)]);
disp(['Validation Time KNN: ', num2str(validationTime)]);
disp(['Testing Time KNN: ', num2str(executionTime)]);
disp(' ');