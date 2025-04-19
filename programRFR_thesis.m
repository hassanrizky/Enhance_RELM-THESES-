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

% Train Random Forest Regression model
tic;
modelRF = fitrensemble(X_train, Y_train, 'Method', 'Bag', 'NumLearningCycles', 100, 'Learners', 'Tree');
trainingTime = toc;

% Validate the model
tic;
Y_ValTest = predict(modelRF, X_val);
validationTime = toc;

rmseVal = rmse(Y_ValTest', Y_val');
maeVal = mae(Y_ValTest', Y_val');
mapeVal = mape(Y_ValTest', Y_val');
rsquareVal = rsquare(Y_ValTest, Y_val);

% Test the model
tic;
Y_OutTest = predict(modelRF, X_test);
executionTime = toc;

rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

% Display results
disp('---------Random Forest Regression----------');
disp('Validation Data');
disp(['RMSE Val RF: ', num2str(rmseVal)]);
disp(['MAE Val RF: ', num2str(maeVal)]);
disp(['MAPE Val RF: ', num2str(mapeVal)]);
disp(['R-Square Val RF: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test RF: ', num2str(rmseTest)]);
disp(['MAE Test RF: ', num2str(maeTest)]);
disp(['MAPE Test RF: ', num2str(mapeTest)]);
disp(['R-Square Test RF: ', num2str(rsquareTest)]);
disp(' ');
disp('Execution Time');
disp(['Training Time RF: ', num2str(trainingTime)]);
disp(['Validation Time RF: ', num2str(validationTime)]);
disp(['Testing Time RF: ', num2str(executionTime)]);
disp(' ');