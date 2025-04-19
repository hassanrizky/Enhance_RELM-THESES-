% Load your dataset or generate synthetic data
% Example:
% rtt_data = load('output_combined_asli - output_combined_asli.csv');
% X = rtt_data.features; % Input features
% y = rtt_data.labels;   % Target labels

% For demonstration purposes, let's generate synthetic data
% rng('default'); % For reproducibility
% X = randn(100, 5); % 100 samples with 5 features
% y = 2*X(:, 1) + 3*X(:, 2) - 1.5*X(:, 3) + randn(100, 1); % Linear relationship with noise

% Load data dari file CSV
filename = 'output_combined_asli - Copy of output_combined_asli.csv';
data = readtable(filename);

%%
% Pilih fitur yang akan digunakan
X = data{:, 1:end-1}; % Ambil semua kolom kecuali kolom terakhir sebagai fitur
y = data{:, end};     % Ambil kolom terakhir sebagai label
%%
% Split the dataset into training and testing sets
train_ratio = 0.8; % 80% for training, 20% for testing
num_samples = size(X, 1);
num_train = round(train_ratio * num_samples);

X_train = X(1:num_train, :);
y_train = y(1:num_train);

X_test = X(num_train+1:end, :);
y_test = y(num_train+1:end);

% Set parameters for RELM
hidden_layer_size = 50;
C = 1; % Regularization parameter

% Train RELM
input_size = size(X_train, 2);
input_weights = randn(hidden_layer_size, input_size);
bias = randn(hidden_layer_size, 1);

% Calculate hidden layer output
H_train = tanh(input_weights*X_train' + repmat(bias, 1, num_train));

% Regularization matrix
regularization_matrix = eye(hidden_layer_size)/C;

% Calculate output weights using Moore-Penrose generalized inverse
H_train_numeric = double(H_train);
output_weights = pinv(H_train_numeric' * H_train_numeric + regularization_matrix) * (H_train_numeric' * y_train);

% Test RELM on the testing set
H_test = tanh(input_weights*X_test' + repmat(bias, 1, size(X_test, 1)));
y_pred = H_test' * output_weights;

% Evaluate performance
mse = mean((y_test - y_pred).^2);
fprintf('Mean Squared Error on Test Set: %.4f\n', mse);

% Plot results
figure;
plot(y_test, 'b', 'DisplayName', 'Actual RTT');
hold on;
plot(y_pred, 'r', 'DisplayName', 'Predicted RTT');
xlabel('Sample');
ylabel('RTT Value');
title('Actual vs. Predicted RTT');
legend('show');
grid on;