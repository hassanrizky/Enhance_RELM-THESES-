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

% Train Lasso Regression model
tic;
modelLasso = fitrlinear(X_train, Y_train, 'Learner', 'leastsquares', 'Regularization', 'lasso');
trainingTime = toc;

% Validate the model
tic;
Y_ValTest = predict(modelLasso, X_val);
validationTime = toc;

rmseVal = rmse(Y_ValTest', Y_val');
maeVal = mae(Y_ValTest', Y_val');
mapeVal = mape(Y_ValTest', Y_val');
rsquareVal = rsquare(Y_ValTest, Y_val);

% Test the model
tic;
Y_OutTest = predict(modelLasso, X_test);
executionTime = toc;

rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

% Display results
disp('---------Lasso Regression----------');
disp('Validation Data');
disp(['RMSE Val Lasso: ', num2str(rmseVal)]);
disp(['MAE Val Lasso: ', num2str(maeVal)]);
disp(['MAPE Val Lasso: ', num2str(mapeVal)]);
disp(['R-Square Val Lasso: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test Lasso: ', num2str(rmseTest)]);
disp(['MAE Test Lasso: ', num2str(maeTest)]);
disp(['MAPE Test Lasso: ', num2str(mapeTest)]);
disp(['R-Square Test Lasso: ', num2str(rsquareTest)]);
disp(' ');
disp('Execution Time');
disp(['Training Time Lasso: ', num2str(trainingTime)]);
disp(['Validation Time Lasso: ', num2str(validationTime)]);
disp(['Testing Time Lasso: ', num2str(executionTime)]);
disp(' ');