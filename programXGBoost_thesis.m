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

% Convert data to DMatrix format
dtrain = xgboost.DMatrix(X_train, label=Y_train);
dval = xgboost.DMatrix(X_val, label=Y_val);
dtest = xgboost.DMatrix(X_test, label=Y_test);

% Set parameters for XGBoost
params = struct();
params.objective = 'reg:squarederror'; % Regression with squared error
params.eta = 0.1; % Learning rate
params.max_depth = 6; % Maximum depth of a tree
params.subsample = 0.8; % Subsample ratio of the training instances
params.colsample_bytree = 0.8; % Subsample ratio of columns when constructing each tree
params.eval_metric = 'rmse'; % Evaluation metric

% Train the model
num_round = 100; % Number of boosting rounds
evals = {dval, 'eval'};
tic;
modelXGB = xgboost.train(params, dtrain, num_round, evals);
trainingTime = toc;

% Validate the model
tic;
Y_ValTest = modelXGB.predict(dval);
validationTime = toc;

rmseVal = rmse(Y_ValTest', Y_val');
maeVal = mae(Y_ValTest', Y_val');
mapeVal = mape(Y_ValTest', Y_val');
rsquareVal = rsquare(Y_ValTest, Y_val);

% Test the model
tic;
Y_OutTest = modelXGB.predict(dtest);
executionTime = toc;

rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

% Display results
disp('---------XGBoost Regression----------');
disp('Validation Data');
disp(['RMSE Val XGB: ', num2str(rmseVal)]);
disp(['MAE Val XGB: ', num2str(maeVal)]);
disp(['MAPE Val XGB: ', num2str(mapeVal)]);
disp(['R-Square Val XGB: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test XGB: ', num2str(rmseTest)]);
disp(['MAE Test XGB: ', num2str(maeTest)]);
disp(['MAPE Test XGB: ', num2str(mapeTest)]);
disp(['R-Square Test XGB: ', num2str(rsquareTest)]);
disp(' ');
disp('Execution Time');
disp(['Training Time XGB: ', num2str(trainingTime)]);
disp(['Validation Time XGB: ', num2str(validationTime)]);
disp(['Testing Time XGB: ', num2str(executionTime)]);
disp(' ');