% Data Contoh
% Data ini hanya contoh. Anda perlu mengumpulkan data RTT, kondisi jaringan, dll.
% Fitur-fitur dalam RTT samples:
% 1. RTT sebelumnya (ms)
% 2. Jumlah retransmisi
% 3. Variasi RTT (ms)
rtt_samples = [
    50, 2, 0.1;
    55, 2, 0.1;
    60, 3, 0.2;
    70, 3, 0.2;
    65, 3, 0.2;
    75, 4, 0.3;
    80, 4, 0.3;
    85, 5, 0.4;
    90, 5, 0.4;
    95, 6, 0.5;
    100, 6, 0.5;
    110, 7, 0.6
];

% Target values (RTT to predict)
rtt_targets = [
    52;
    58;
    63;
    72;
    68;
    77;
    83;
    88;
    93;
    98;
    104;
    113
];

% Membagi data menjadi data pelatihan dan pengujian
cv = cvpartition(size(rtt_samples, 1), 'HoldOut', 0.2);
idx = cv.test;

% Data pelatihan
X_train = rtt_samples(~idx, :);
y_train = rtt_targets(~idx, :);

% Data pengujian
X_test = rtt_samples(idx, :);
y_test = rtt_targets(idx, :);

% Parameter model RELM
n_hidden = 100; 
lambda = 1; 
%activation_function = @(x) 1./(1 + exp(-x)); % Fungsi aktivasi sigmoid
activation_function = @(x) max(0, x); % Fungsi Aktivasi RelU

% Membuat dan melatih model RELM
[W, b, beta] = train_relm(X_train, y_train, n_hidden, lambda, activation_function);

% Memprediksi nilai RTT pada data pengujian
y_pred = predict_relm(X_test, W, b, beta, activation_function);

% Menghitung Mean Squared Error (MSE)
mseTest = mean((y_test - y_pred).^2);
rmseTest = rmse(y_pred', y_test');
maeTest = mae(y_pred', y_test');
mapeTest = mape(y_pred', y_test');
%rsquareTest = rsquare(y_pred, y_test);

fprintf('Mean Squared Error: %.2f\n', mseTest);
fprintf('Root Mean Squared Error: %.2f\n', rmseTest);
fprintf('Mean Absolute Error: %.2f\n', maeTest);
fprintf('Mean Absolute Percentage Error: %.2f\n', mapeTest);
%fprintf('R2: %.2f\n', rsquareTest);

% Contoh prediksi untuk data baru
new_data = [85, 5, 0.4];  % Sesuaikan dengan fitur data baru
predicted_rtt = predict_relm(new_data, W, b, beta, activation_function);
fprintf('Predicted RTT: %.2f ms\n', predicted_rtt);



% Menghitung Timeout
predicted_rttvar = std(y_train); % Variasi RTT dihitung dari data pelatihan
timeout_interval = calculate_timeout(predicted_rtt, predicted_rttvar);
fprintf('Calculated Timeout Interval: %.2f ms\n', timeout_interval);

% Fungsi untuk menghitung Timeout berdasarkan RTT dan Variasi RTT
function timeout_interval = calculate_timeout(rtt, rttvar, g, k)
    if nargin < 3
        g = 1.0;
    end
    if nargin < 4
        k = 4.0;
    end
    timeout_interval = rtt + max(g, k * rttvar);
end

% Fungsi untuk melatih model RELM
function [W, b, beta] = train_relm(X, y, n_hidden, lambda, activation_function)
    input_size = size(X, 2);
    
    % Inisialisasi bobot dan bias acak
    W = rand(input_size, n_hidden) * 2 - 1;  % Bobot input
    b = rand(1, n_hidden) * 2 - 1;           % Bias
    
    % Hitung keluaran lapisan tersembunyi
    H = activation_function(X * W + b);
    
    % Regularisasi dan pelatihan
    beta = (H' * H + lambda * eye(n_hidden)) \ (H' * y);
end

% Fungsi untuk memprediksi dengan model RELM
function y_pred = predict_relm(X, W, b, beta, activation_function)
    H = activation_function(X * W + b);
    y_pred = H * beta;
end

%%
