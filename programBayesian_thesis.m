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

% Train Bayesian Ridge Regression model
tic;
modelBayesian = fitrlinear(X_train, Y_train, 'Learner', 'bayesianridge');
trainingTime = toc;

% Validate the model
tic;
Y_ValTest = predict(modelBayesian, X_val);
validationTime = toc;

rmseVal = rmse(Y_ValTest', Y_val');
maeVal = mae(Y_ValTest', Y_val');
mapeVal = mape(Y_ValTest', Y_val');
rsquareVal = rsquare(Y_ValTest, Y_val);

% Test the model
tic;
Y_OutTest = predict(modelBayesian, X_test);
executionTime = toc;

rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

% Display results
disp('---------Bayesian Ridge Regression----------');
disp('Validation Data');
disp(['RMSE Val Bayesian: ', num2str(rmseVal)]);
disp(['MAE Val Bayesian: ', num2str(maeVal)]);
disp(['MAPE Val Bayesian: ', num2str(mapeVal)]);
disp(['R-Square Val Bayesian: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test Bayesian: ', num2str(rmseTest)]);
disp(['MAE Test Bayesian: ', num2str(maeTest)]);
disp(['MAPE Test Bayesian: ', num2str(mapeTest)]);
disp(['R-Square Test Bayesian: ', num2str(rsquareTest)]);
disp(' ');
disp('Execution Time');
disp(['Training Time Bayesian: ', num2str(trainingTime)]);
disp(['Validation Time Bayesian: ', num2str(validationTime)]);
disp(['Testing Time Bayesian: ', num2str(executionTime)]);
disp(' ');

%%
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

% Train Ridge Regression model
tic;
modelRidge = fitrlinear(X_train, Y_train, 'Learner', 'leastsquares', 'Regularization', 'bayesianridge');
trainingTime = toc;

% Validate the model
tic;
Y_ValTest = predict(modelRidge, X_val);
validationTime = toc;

rmseVal = rmse(Y_ValTest', Y_val');
maeVal = mae(Y_ValTest', Y_val');
mapeVal = mape(Y_ValTest', Y_val');
rsquareVal = rsquare(Y_ValTest, Y_val);

% Test the model
tic;
Y_OutTest = predict(modelRidge, X_test);
executionTime = toc;

rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

% Display results
disp('---------Ridge Regression----------');
disp('Validation Data');
disp(['RMSE Val Ridge: ', num2str(rmseVal)]);
disp(['MAE Val Ridge: ', num2str(maeVal)]);
disp(['MAPE Val Ridge: ', num2str(mapeVal)]);
disp(['R-Square Val Ridge: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test Ridge: ', num2str(rmseTest)]);
disp(['MAE Test Ridge: ', num2str(maeTest)]);
disp(['MAPE Test Ridge: ', num2str(mapeTest)]);
disp(['R-Square Test Ridge: ', num2str(rsquareTest)]);
disp(' ');
disp('Execution Time');
disp(['Training Time Ridge: ', num2str(trainingTime)]);
disp(['Validation Time Ridge: ', num2str(validationTime)]);
disp(['Testing Time Ridge: ', num2str(executionTime)]);
disp(' ');