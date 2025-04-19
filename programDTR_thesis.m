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

% Train Decision Tree Regression model
tic;
modelTree = fitrtree(X_train, Y_train);
trainingTime = toc;

% Validate the model
tic;
Y_ValTest = predict(modelTree, X_val);
validationTime = toc;

rmseVal = rmse(Y_ValTest', Y_val');
maeVal = mae(Y_ValTest', Y_val');
mapeVal = mape(Y_ValTest', Y_val');
rsquareVal = rsquare(Y_ValTest, Y_val);

% Test the model
tic;
Y_OutTest = predict(modelTree, X_test);
executionTime = toc;

rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

% Display results
disp('---------Decision Tree Regression----------');
disp('Validation Data');
disp(['RMSE Val Tree: ', num2str(rmseVal)]);
disp(['MAE Val Tree: ', num2str(maeVal)]);
disp(['MAPE Val Tree: ', num2str(mapeVal)]);
disp(['R-Square Val Tree: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test Tree: ', num2str(rmseTest)]);
disp(['MAE Test Tree: ', num2str(maeTest)]);
disp(['MAPE Test Tree: ', num2str(mapeTest)]);
disp(['R-Square Test Tree: ', num2str(rsquareTest)]);
disp(' ');
disp('Execution Time');
disp(['Training Time Tree: ', num2str(trainingTime)]);
disp(['Validation Time Tree: ', num2str(validationTime)]);
disp(['Testing Time Tree: ', num2str(executionTime)]);
disp(' ');