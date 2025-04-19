%%L1-Norm RELM

% Load the data
load('data_delay.mat')

% Assume data is already loaded and preprocessed
X = data(:, 2:101);
Y = data(:, 1);

% Splitting data into train, validation, and test sets without shuffling
trainIndex = 1:600;
validationIndex = 601:700;
testIndex = 701:1000;

X_train = X(trainIndex, :);
Y_train = Y(trainIndex, 1);

X_val = X(validationIndex, :);
Y_val = Y(validationIndex, 1);

X_test = X(testIndex, :);
Y_test = Y(testIndex, 1);

%MODELRELM
% Set parameters for RELM
hidden_layer_size = 100;
lambda = 0.00001; % Regularization parameter for L1-norm

% Train RELM (DASAR YANG DIGUNAKAN)
input_size = size(X_train, 2);

% Initialize weights and bias using He initialization for ReLU
input_weights = randn(hidden_layer_size, input_size) * sqrt(2 / input_size);
bias = randn(hidden_layer_size, 1) * sqrt(2 / input_size);

% Calculate hidden layer output
num_train = size(X_train, 1);
H_train = max(0, input_weights * X_train' + repmat(bias, 1, num_train)); % ReLU

% L1-regularized least squares solution for output weights
H_train_numeric = double(H_train');

cvx_begin
    variable output_weights(hidden_layer_size)
    minimize(norm(H_train_numeric * output_weights - Y_train, 2) + lambda * norm(output_weights, 1))
cvx_end

% Measure training time
trainingTime = toc;

% Test RELM on the validation set
H_val = max(0, input_weights * X_val' + repmat(bias, 1, size(X_val, 1))); % ReLU
Y_ValTest = H_val' * output_weights;
validationTime = toc;

% Compute validation metrics
rmseVal = sqrt(mean((Y_ValTest' - Y_val).^2));
maeVal = mean(abs(Y_ValTest' - Y_val));
mapeVal = mean(abs((Y_ValTest' - Y_val) ./ Y_val)) * 100;
rsquareVal = 1 - sum((Y_ValTest' - Y_val).^2) / sum((Y_val - mean(Y_val)).^2);

% Test RELM on the testing set
H_test = max(0, input_weights * X_test' + repmat(bias, 1, size(X_test, 1))); % ReLU
Y_OutTest = H_test' * output_weights;
executionTime = toc;

% Compute test metrics
rmseTest = sqrt(mean((Y_OutTest' - Y_test).^2));
maeTest = mean(abs(Y_OutTest' - Y_test));
mapeTest = mean(abs((Y_OutTest' - Y_test) ./ Y_test)) * 100;
rsquareTest = 1 - sum((Y_OutTest' - Y_test).^2) / sum((Y_test - mean(Y_test)).^2);

% Display results
disp('---------L1-norm RELM----------');
disp('Validation Data');
disp(['RMSE Val RELM: ', num2str(rmseVal)]);
disp(['MAE Val RELM: ', num2str(maeVal)]);
disp(['MAPE Val RELM: ', num2str(mapeVal)]);
disp(['R-Square Val RELM: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test RELM: ', num2str(rmseTest)]);
disp(['MAE Test RELM: ', num2str(maeTest)]);
disp(['MAPE Test RELM: ', num2str(mapeTest)]);
disp(['R-Square Test RELM: ', num2str(rsquareTest)]);
disp(' ');
disp('Execution Time');
disp(['Training Time RELM: ', num2str(trainingTime)]);
disp(['Validation Time RELM: ', num2str(validationTime)]);
disp(['Testing Time RELM: ', num2str(executionTime)]);
disp(' ');

%% Estimation using MGU

% Load data
X = data(:, 2:101);
Y = data(:, 1);

% Split data into training, validation, and test sets
X_train = X(1:600, :);
Y_train = Y(1:600, :);
X_val = X(601:700, :);
Y_val = Y(601:700, :);
X_test = X(701:1000, :);
Y_test = Y(701:1000, :);

% Set parameters
hidden_layer_size = 100;
C = 0.00000000001;

% Train MGU-RELM model
tic;
[Y_val_pred, output_weights, trainingTime] = trainMGU(X_train, Y_train, X_val, Y_val, hidden_layer_size, C);

% Calculate validation metrics
rmseVal = sqrt(mean((Y_val_pred - Y_val).^2));
maeVal = mean(abs(Y_val_pred - Y_val));
mapeVal = mean(abs((Y_val_pred - Y_val) ./ Y_val)) * 100;
rsquareVal = 1 - sum((Y_val_pred - Y_val).^2) / sum((Y_val - mean(Y_val)).^2);

% Test MGU-RELM model
num_test = size(X_test, 1);
H_test = zeros(hidden_layer_size, num_test);
for t = 1:num_test
    x_t = X_test(t, :)';
    if t == 1
        h_t_minus_1 = zeros(hidden_layer_size, 1);
    else
        h_t_minus_1 = H_test(:, t-1);
    end
    
    z_t = sigmoid(Wz * x_t + Uz * h_t_minus_1 + bias);
    r_t = sigmoid(Wr * x_t + Ur * h_t_minus_1 + bias);
    h_hat_t = tanh(Wh * x_t + Uh * (r_t .* h_t_minus_1) + bias);
    H_test(:, t) = (1 - z_t) .* h_t_minus_1 + z_t .* h_hat_t;
end

Y_test_pred = H_test' * output_weights;

% Calculate test metrics
rmseTest = sqrt(mean((Y_test_pred - Y_test).^2));
maeTest = mean(abs(Y_test_pred - Y_test));
mapeTest = mean(abs((Y_test_pred - Y_test) ./ Y_test)) * 100;
rsquareTest = 1 - sum((Y_test_pred - Y_test).^2) / sum((Y_test - mean(Y_test)).^2);

% Display results
disp('---------MGU-RELM----------');
disp('Validation Data');
disp(['RMSE Val: ', num2str(rmseVal)]);
disp(['MAE Val: ', num2str(maeVal)]);
disp(['MAPE Val: ', num2str(mapeVal)]);
disp(['R-Square Val: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test: ', num2str(rmseTest)]);
disp(['MAE Test: ', num2str(maeTest)]);
disp(['MAPE Test: ', num2str(mapeTest)]);
disp(['R-Square Test: ', num2str(rsquareTest)]);
disp(' ');
disp('Execution Time');
disp(['Training Time: ', num2str(trainingTime)]);

function [Y_pred, output_weights, trainingTime] = trainMGU(X_train, Y_train, X_val, Y_val, hidden_layer_size, C)
    % Initialize parameters
    input_size = size(X_train, 2);
    num_train = size(X_train, 1);
    
    % Initialize weights and biases
    input_weights = randn(hidden_layer_size, input_size) * sqrt(2 / input_size);
    bias = randn(hidden_layer_size, 1) * sqrt(2 / input_size);
    
    % Initialize MGU parameters
    Wz = randn(hidden_layer_size, input_size) * sqrt(2 / input_size); % Update gate
    Uz = randn(hidden_layer_size, hidden_layer_size) * sqrt(2 / hidden_layer_size);
    Wr = randn(hidden_layer_size, input_size) * sqrt(2 / input_size); % Reset gate
    Ur = randn(hidden_layer_size, hidden_layer_size) * sqrt(2 / hidden_layer_size);
    Wh = randn(hidden_layer_size, input_size) * sqrt(2 / input_size); % Candidate activation
    Uh = randn(hidden_layer_size, hidden_layer_size) * sqrt(2 / hidden_layer_size);
    
    % Forward pass for MGU
    H_train = zeros(hidden_layer_size, num_train);
    for t = 1:num_train
        x_t = X_train(t, :)';
        if t == 1
            h_t_minus_1 = zeros(hidden_layer_size, 1);
        else
            h_t_minus_1 = H_train(:, t-1);
        end
        
        z_t = sigmoid(Wz * x_t + Uz * h_t_minus_1 + bias);
        r_t = sigmoid(Wr * x_t + Ur * h_t_minus_1 + bias);
        h_hat_t = tanh(Wh * x_t + Uh * (r_t .* h_t_minus_1) + bias);
        H_train(:, t) = (1 - z_t) .* h_t_minus_1 + z_t .* h_hat_t;
    end
    
    % Regularization matrix
    regularization_matrix = eye(hidden_layer_size) / C;
    
    % Calculate output weights using Moore-Penrose generalized inverse
    H_train_numeric = double(H_train');
    output_weights = pinv(H_train_numeric' * H_train_numeric + regularization_matrix) * (H_train_numeric' * Y_train);
    
    % Validate MGU
    num_val = size(X_val, 1);
    H_val = zeros(hidden_layer_size, num_val);
    for t = 1:num_val
        x_t = X_val(t, :)';
        if t == 1
            h_t_minus_1 = zeros(hidden_layer_size, 1);
        else
            h_t_minus_1 = H_val(:, t-1);
        end
        
        z_t = sigmoid(Wz * x_t + Uz * h_t_minus_1 + bias);
        r_t = sigmoid(Wr * x_t + Ur * h_t_minus_1 + bias);
        h_hat_t = tanh(Wh * x_t + Uh * (r_t .* h_t_minus_1) + bias);
        H_val(:, t) = (1 - z_t) .* h_t_minus_1 + z_t .* h_hat_t;
    end
    
    Y_pred = H_val' * output_weights;
    trainingTime = toc;
end

function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end

%% 

% Load data
X = data(:, 2:101);
Y = data(:, 1);

% Set random seed for reproducibility
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

% Set parameters for MGU
hidden_layer_size = 100;
C = 0.000000001;

% Initialize weights and biases
input_size = size(X_train, 2);
bias = randn(hidden_layer_size, 1) * sqrt(2 / input_size);
Wz = randn(hidden_layer_size, input_size) * sqrt(2 / input_size); % Update gate
Uz = randn(hidden_layer_size, hidden_layer_size) * sqrt(2 / hidden_layer_size);
Wr = randn(hidden_layer_size, input_size) * sqrt(2 / input_size); % Reset gate
Ur = randn(hidden_layer_size, hidden_layer_size) * sqrt(2 / hidden_layer_size);
Wh = randn(hidden_layer_size, input_size) * sqrt(2 / input_size); % Candidate activation
Uh = randn(hidden_layer_size, hidden_layer_size) * sqrt(2 / hidden_layer_size);

tic;
% Train MGU
num_train = size(X_train, 1);
H_train = zeros(hidden_layer_size, num_train);
for t = 1:num_train
    x_t = X_train(t, :)';
    if t == 1
        h_t_minus_1 = zeros(hidden_layer_size, 1);
    else
        h_t_minus_1 = H_train(:, t-1);
    end
    
    z_t = sigmoid(Wz * x_t + Uz * h_t_minus_1 + bias);
    r_t = sigmoid(Wr * x_t + Ur * h_t_minus_1 + bias);
    h_hat_t = tanh(Wh * x_t + Uh * (r_t .* h_t_minus_1) + bias);
    H_train(:, t) = (1 - z_t) .* h_t_minus_1 + z_t .* h_hat_t;
end

% Regularization matrix
regularization_matrix = eye(hidden_layer_size) / C;

% Calculate output weights using Moore-Penrose generalized inverse
H_train_numeric = double(H_train');
output_weights = pinv(H_train_numeric' * H_train_numeric + regularization_matrix) * (H_train_numeric' * Y_train);
trainingTime = toc;

tic;
% Test MGU on the Validation set
num_val = size(X_val, 1);
H_val = zeros(hidden_layer_size, num_val);
for t = 1:num_val
    x_t = X_val(t, :)';
    if t == 1
        h_t_minus_1 = zeros(hidden_layer_size, 1);
    else
        h_t_minus_1 = H_val(:, t-1);
    end
    
    z_t = sigmoid(Wz * x_t + Uz * h_t_minus_1 + bias);
    r_t = sigmoid(Wr * x_t + Ur * h_t_minus_1 + bias);
    h_hat_t = tanh(Wh * x_t + Uh * (r_t .* h_t_minus_1) + bias);
    H_val(:, t) = (1 - z_t) .* h_t_minus_1 + z_t .* h_hat_t;
end

Y_ValTest = H_val' * output_weights;
validationTime = toc;

rmseVal = rmse(Y_ValTest', Y_val');
maeVal = mae(Y_ValTest', Y_val');
mapeVal = mape(Y_ValTest', Y_val');
rsquareVal = rsquare(Y_ValTest, Y_val);

tic;
% Test MGU on the testing set
num_test = size(X_test, 1);
H_test = zeros(hidden_layer_size, num_test);
for t = 1:num_test
    x_t = X_test(t, :)';
    if t == 1
        h_t_minus_1 = zeros(hidden_layer_size, 1);
    else
        h_t_minus_1 = H_test(:, t-1);
    end
    
    z_t = sigmoid(Wz * x_t + Uz * h_t_minus_1 + bias);
    r_t = sigmoid(Wr * x_t + Ur * h_t_minus_1 + bias);
    h_hat_t = tanh(Wh * x_t + Uh * (r_t .* h_t_minus_1) + bias);
    H_test(:, t) = (1 - z_t) .* h_t_minus_1 + z_t .* h_hat_t;
end

Y_OutTest = H_test' * output_weights;
executionTime = toc;

rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

disp('---------MGU----------');
disp('Validation Data');
disp(['RMSE Val MGU: ', num2str(rmseVal)]);
disp(['MAE Val MGU: ', num2str(maeVal)]);
disp(['MAPE Val MGU: ', num2str(mapeVal)]);
disp(['R-Square Val MGU: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test MGU: ', num2str(rmseTest)]);
disp(['MAE Test MGU: ', num2str(maeTest)]);
disp(['MAPE Test MGU: ', num2str(mapeTest)]);
disp(['R-Square Test MGU: ', num2str(rsquareTest)]);
disp(' ');
disp('Execution Time');
disp(['Training Time MGU: ', num2str(trainingTime)]);
disp(['Validation Time MGU: ', num2str(validationTime)]);
disp(['Testing Time MGU: ', num2str(executionTime)]);
disp(' ');



% Define RMSE function
function error = rmse(predictions, targets)
    error = sqrt(mean((predictions - targets).^2));
end

% Define MAE function
function error = mae(predictions, targets)
    error = mean(abs(predictions - targets));
end

% Define MAPE function
function error = mape(predictions, targets)
    error = mean(abs((predictions - targets) ./ targets)) * 100;
end

% Define R-Square function
function rsq = rsquare(predictions, targets)
    ss_res = sum((targets - predictions).^2);
    ss_tot = sum((targets - mean(targets)).^2);
    rsq = 1 - (ss_res / ss_tot);
end

% Define sigmoid function at the end of the script
function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end
