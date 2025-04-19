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

% Normalize data
[X_train, X_mu, X_sigma] = zscore(X_train);
X_val = (X_val - X_mu) ./ X_sigma;
X_test = (X_test - X_mu) ./ X_sigma;

% Create and train Neural Network model
hiddenLayerSize = 100; % Number of neurons in the hidden layer
net = fitnet(hiddenLayerSize);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 600 / 1000;
net.divideParam.valRatio = 100 / 1000;
net.divideParam.testRatio = 300 / 1000;

% Train the network
tic;
[net, tr] = train(net, X_train', Y_train');
trainingTime = toc;

% Validate the model
tic;
Y_ValTest = net(X_val');
validationTime = toc;

rmseVal = rmse(Y_ValTest', Y_val');
maeVal = mae(Y_ValTest', Y_val');
mapeVal = mape(Y_ValTest', Y_val');
rsquareVal = rsquare(Y_ValTest, Y_val);

% Test the model
tic;
Y_OutTest = net(X_test');
executionTime = toc;

rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

% Display results
disp('---------Neural Network Regression----------');
disp('Validation Data');
disp(['RMSE Val NN: ', num2str(rmseVal)]);
disp(['MAE Val NN: ', num2str(maeVal)]);
disp(['MAPE Val NN: ', num2str(mapeVal)]);
%disp(['R-Square Val NN: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test NN: ', num2str(rmseTest)]);
disp(['MAE Test NN: ', num2str(maeTest)]);
disp(['MAPE Test NN: ', num2str(mapeTest)]);
%disp(['R-Square Test NN: ', num2str(rsquareTest)]);
disp(' ');
disp('Execution Time');
disp(['Training Time NN: ', num2str(trainingTime)]);
disp(['Validation Time NN: ', num2str(validationTime)]);
disp(['Testing Time NN: ', num2str(executionTime)]);
disp(' ');