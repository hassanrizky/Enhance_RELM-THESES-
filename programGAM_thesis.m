X = data(:, 2:100);
Y = data(:, 101);

rng(82)
index00 = randperm(1000);

trainIndex = index00(1:600);
validationIndex = index00(601:700);
testIndex = index00(701:1000);

X_train = X(trainIndex, :);
Y_train = Y(trainIndex, 1);

X_val = X(validationIndex, :);
Y_val = Y(validationIndex, 1);

X_test = X(testIndex, :);
Y_test = Y(testIndex, 1);

tic;
modelGAM = fitrgam(X_train, Y_train);
trainingTime = toc;

tic;
Y_ValTest = predict(modelGAM, X_val);
validationTime = toc;

rmseVal = rmse(Y_ValTest', Y_val');
maeVal = mae(Y_ValTest', Y_val');
mapeVal = mape(Y_ValTest', Y_val');
rsquareVal = rsquare(Y_ValTest, Y_val);

tic;
Y_OutTest = predict(modelGAM, X_test);
executionTime = toc;

rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

disp('---------GAM----------');
disp('Validation Data');
disp(['RMSE Val GAM: ', num2str(rmseVal)]);
disp(['MAE Val GAM: ', num2str(maeVal)]);
disp(['MAPE Val GAM: ', num2str(mapeVal)]);
disp(['R-Square Val GAM: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test GAM: ', num2str(rmseTest)]);
disp(['MAE Test GAM: ', num2str(maeTest)]);
disp(['MAPE Test GAM: ', num2str(mapeTest)]);
disp(['R-Square Test GAM: ', num2str(rsquareTest)]);
disp(' ');
disp('Execution Time');
disp(['Training Time GAM: ', num2str(trainingTime)]);
disp(['Validation Time GAM: ', num2str(validationTime)]);
disp(['Testing Time GAM: ', num2str(executionTime)]);
disp(' ');

%%
% Assuming X and Y are already defined

k = 5; % Number of folds
indices = crossvalind('Kfold', size(X, 1), k);

rmseValues = zeros(1, k);
maeValues = zeros(1, k);
mapeValues = zeros(1, k);
rsquareValues = zeros(1, k);

for fold = 1:k
    testIndices = (indices == fold);
    trainIndices = ~testIndices;

    X_train = X(trainIndices, :);
    Y_train = Y(trainIndices, 1);

    X_test = X(testIndices, :);
    Y_test = Y(testIndices, 1);

    modelGAM = fitrgam(X_train, Y_train);
    Y_OutTest = predict(modelGAM, X_test);

    rmseValues(fold) = rmse(Y_OutTest', Y_test');
    maeValues(fold) = mae(Y_OutTest', Y_test');
    mapeValues(fold) = mape(Y_OutTest', Y_test');
    rsquareValues(fold) = rsquare(Y_OutTest, Y_test);
end

% Calculate average performance metrics over all folds
avgRMSE = mean(rmseValues);
avgMAE = mean(maeValues);
avgMAPE = mean(mapeValues);
avgRSquare = mean(rsquareValues);

% Display or use the average metrics as needed
disp(['Average RMSE: ', num2str(avgRMSE)]);
disp(['Average MAE: ', num2str(avgMAE)]);
disp(['Average MAPE: ', num2str(avgMAPE)]);
disp(['Average R-squared: ', num2str(avgRSquare)]);