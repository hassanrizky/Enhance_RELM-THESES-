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
%MODELRELM
% Set parameters for RELM
hidden_layer_size = 100;
C = 0.000000001;

% Train RELM (DASAR YANG DIGUNAKAN)
input_size = size(X_train, 2);
%input_weights = randn(hidden_layer_size, input_size);
%bias = randn(hidden_layer_size, 1);

% Inisialisasi bobot dan bias dengan inisialisasi He untuk relU
input_weights = randn(hidden_layer_size, input_size) * sqrt(2 / input_size);
bias = randn(hidden_layer_size, 1) * sqrt(2 / input_size);

% Inisialisasi bobot dan bias dengan inisialisasi untuk tanh atau sigmoid
%input_weights = randn(hidden_layer_size, input_size) * sqrt(1 / input_size);
%bias = randn(hidden_layer_size, 1) * sqrt(1 / input_size);

% Calculate hidden layer output
num_train = size(X_train, 1);
%H_train = tanh(input_weights*X_train' + repmat(bias, 1, num_train)); %tanh
H_train = max(0, input_weights * X_train' + repmat(bias, 1, num_train)); %relu
%H_train = 1 ./ (1 + exp(-(input_weights * X_train' + repmat(bias, 1, num_train)))); % sigmoid

% Regularization matrix
regularization_matrix = eye(hidden_layer_size)/C;

% Calculate output weights using Moore-Penrose generalized inverse
H_train_numeric = double(H_train');
output_weights = pinv(H_train_numeric' * H_train_numeric + regularization_matrix) * (H_train_numeric' *Y_train);
trainingTime = toc;

tic;
% Test RELM on the Validaton set
%H_val = tanh(input_weights*X_val' + repmat(bias, 1, size(X_val, 1))); %tanh
H_val = max(0, input_weights * X_val' + repmat(bias, 1, size(X_val,1))); %relu
%H_val = 1 ./ (1 + exp(-(input_weights * X_val' + repmat(bias, 1, size(X_val, 1))))); %sigmoid
Y_ValTest = H_val' * output_weights;
validationTime = toc;

rmseVal = rmse(Y_ValTest', Y_val');
maeVal = mae(Y_ValTest', Y_val');
mapeVal = mape(Y_ValTest', Y_val');
rsquareVal = rsquare(Y_ValTest, Y_val);

tic;
% Test RELM on the testing set
%H_test = tanh(input_weights*X_test' + repmat(bias, 1, size(X_test, 1))); %tanh
H_test = max(0, input_weights * X_test' + repmat(bias, 1, size(X_test,1))); %relu
%H_test = 1 ./ (1 + exp(-(input_weights * X_test' + repmat(bias, 1, size(X_test, 1))))); %sigmoid
Y_OutTest = H_test' * output_weights;
executionTime = toc;

rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

disp('---------RELM----------');
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

% Test RELM on the New Data
%new_data = X_test(1, :);  % Sesuaikan dengan fitur data baru
sample_rtt = X(1, :);
% Parameter
alpha = 0.125;
beta = 0.25;
% Inisialisasi EstimatedRTT dan RTTVar
% Hitung EstimatedRTT
%estimated_rtt = (1 - alpha) * estimated_rtt + alpha * rtt_sample;
%H_new = max(0, input_weights * sample_rtt' + repmat(bias, 1, size(sample_rtt,1))); %relu
%estimated_rtt = H_new' * output_weights;
%estimate_rtt_input = estimated_rtt';
estimated_rtt = sample_rtt(1);  % Anggap nilai awal adalah sampel pertama
rttvar = 0;

% Loop melalui sample RTT untuk memperbarui EstimatedRTT dan RTTVar
for i = 2:length(sample_rtt)
    rtt_sample = sample_rtt(i);
    
    % Hitung RTTVar
    rttvar = (1 - beta) * rttvar + beta * abs(rtt_sample - estimated_rtt);
    
    % Hitung EstimatedRTT
    %estimated_rtt = (1 - alpha) * estimated_rtt + alpha * rtt_sample;
    H_new = max(0, input_weights * rtt_sample' + repmat(bias, 1, size(rtt_sample,1))); %relu
    estimated_rtt = H_new' * output_weights;
    %estimated_rtt_output = estimated_rtt_input(i);

    % Hitung Timeout Interval
    timeout_interval = estimated_rtt + 4 * rttvar;
    
    fprintf('SampleRTT: %.2f ms, EstimatedRTT: %.2f ms, RTTVar: %.2f ms, TimeoutInterval: %.2f ms\n', ...
        rtt_sample, estimated_rtt, rttvar, timeout_interval);
end

% Mean Absolute Error (MAE)
mae = mean(abs(sample_rtt - estimated_rtt));

% Mean Squared Error (MSE)
mse = mean((sample_rtt - estimated_rtt).^2);

% Standard Deviation of Residuals
residuals = sample_rtt - estimated_rtt;
std_residuals = std(residuals);

% Coverage Probability
within_timeout = sum(sample_rtt <= (estimated_rtt + 4 * rttvar));
coverage_probability = within_timeout / length(sample_rtt);

% Percentage of Successful Transmissions
successful_transmissions = sum(sample_rtt <= (estimated_rtt + 4 * rttvar));
success_rate = successful_transmissions / length(sample_rtt);

% Jitter
jitter = std(diff(sample_rtt));

fprintf('MAE: %.2f, MSE: %.2f, Std Residuals: %.2f, Coverage Probability: %.2f, Success Rate: %.2f, Jitter: %.2f\n', ...
    mae, mse, std_residuals, coverage_probability, success_rate, jitter);
%H_val = tanh(input_weights*X_val' + repmat(bias, 1, size(X_val, 1))); %tanh
%H_new = max(0, input_weights * sample_rtt' + repmat(bias, 1, size(X_val,1))); %relu
%H_val = 1 ./ (1 + exp(-(input_weights * X_val' + repmat(bias, 1, size(X_val, 1))))); %sigmoid
%Y_NewPred = H_new' * output_weights;
%fprintf('Predicted RTT: %.2f ms\n', Y_NewPred);

% Menghitung Timeout
%predicted_rttvar = std(Y_train); % Variasi RTT dihitung dari data pelatihan
%timeout_interval = calculate_timeout(Y_NewPred, predicted_rttvar);
%fprintf('Calculated Timeout Interval: %.2f ms\n', timeout_interval);

% Fungsi untuk menghitung Timeout berdasarkan RTT dan Variasi RTT
%function timeout_interval = calculate_timeout(rtt, rttvar, g, k)
%    if nargin < 3
%        g = 1.0;
%    end
%    if nargin < 4
%        k = 4.0;
%    end
%    timeout_interval = rtt + max(g, k * rttvar);
%end


%%
% Plot the original data and the BiLSTM predictions for the first 100 points
%figure;
%plot(1:length(Y_test), Y_test, 'o-', 'DisplayName', 'Original Data');
%hold on;
%plot(1:length(Y_OutTest), Y_OutTest, 'x-', 'DisplayName', 'RELM Predictions');
%xlabel('Delay Value');
%ylabel('RTT Value');
%title('Round Trip TIme Prediction using RELM');
%legend
%hold off;