%% Algoritma proposed untuk nilai C part 2 (Daily Demand
% Dataset)

X = data(:, 1:12);
Y = data(:, 13);

rng(82)
index00 = randperm(60);

trainIndex = index00(1:36);
validationIndex = index00(37:42);
testIndex = index00(43:60);

X_train = X(trainIndex, :);
Y_train = Y(trainIndex, 1);

X_val = X(validationIndex, :);
Y_val = Y(validationIndex, 1);

X_test = X(testIndex, :);
Y_test = Y(testIndex, 1);

tic;
% MODELRELM
% Set parameters for RELM
hidden_layer_size = 100;

% Train RELM (DASAR YANG DIGUNAKAN)
input_size = size(X_train, 2);

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

% Calculate hidden layer output
num_train = size(X_train, 1);
%H_train = max(0, input_weights * X_train' + repmat(bias, 1, num_train)); % relu
H_train_numeric = double(H_train');
trainingTime = toc;

% Inisialisasi variabel untuk pencarian nilai C
C = 1;
best_C = C;
best_rmse_val = Inf;
best_mae_val = Inf;
best_mape_val = Inf;
best_rsquare_val = -Inf;

best_C_forward = C;
best_rmse_val_forward = Inf;
best_mae_val_forward = Inf;
best_mape_val_forward = Inf;
best_rsquare_val_forward = -Inf;

best_C_backward = C;
best_rmse_val_backward = Inf;
best_mae_val_backward = Inf;
best_mape_val_backward = Inf;
best_rsquare_val_backward = -Inf;

% Variabel untuk menghitung perulangan
no_change_forward = 0;
no_change_backward = 0;
stop_forward = false;
stop_backward = false;

disp('Sebelum perulangan');
initial_iteration = true;  % Variabel untuk menandai iterasi pertama
initial_rmse = inf;  % Untuk menyimpan nilai RMSE awal
initial_mae = inf;   % Untuk menyimpan nilai MAE awal
initial_mape = inf;  % Untuk menyimpan nilai MAPE awal
initial_rsquare = -inf; % Untuk menyimpan nilai R-squared awal
while true
    disp('Parameter 1');
    % Evaluasi model dengan nilai C saat ini, hanya jika belum ada perubahan
    if initial_iteration
        disp('Parameter 2');
        [rmseVal, maeVal, mapeVal, rsquareVal] = evaluate_RELM(C, H_train_numeric, Y_train, input_weights, bias, X_val, Y_val);
        
        if initial_rmse == inf
            % Simpan nilai awal saat C = 1
            initial_rmse = rmseVal;
            initial_mae = maeVal;
            initial_mape = mapeVal;
            initial_rsquare = rsquareVal;
        end
        
        if rmseVal < best_rmse_val || (rmseVal == best_rmse_val && maeVal < best_mae_val)
            disp('Parameter 3');
            best_rmse_val = rmseVal;
            best_mae_val = maeVal;
            best_mape_val = mapeVal;
            best_rsquare_val = rsquareVal;
            best_C = C;

            best_rmse_val_forward = rmseVal;
            best_mae_val_forward = maeVal;
            best_mape_val_forward = mapeVal;
            best_rsquare_val_forward = rsquareVal;
            best_C_forward = C;

            best_rmse_val_backward = rmseVal;
            best_mae_val_backward = maeVal;
            best_mape_val_backward = mapeVal;
            best_rsquare_val_backward = rsquareVal;
            best_C_backward = C;
            C_forward = C;
            C_backward = C;
        end
        disp([' C: ', num2str(C)]);
        disp('Validation Data');
        disp(['RMSE Val RELM: ', num2str(rmseVal)]);
        disp(['MAE Val RELM: ', num2str(maeVal)]);
        disp(['MAPE Val RELM: ', num2str(mapeVal)]);
        disp(['R-Square Val RELM: ', num2str(rsquareVal)]);
        initial_iteration = false;
    end
    
    disp('Parameter 7');
    % Evaluasi model dengan nilai C maju jika iterasi maju belum dihentikan
    if ~stop_forward
        disp('Parameter 8');
        C_forward = C_forward * 10;
        [rmseVal_forward, maeVal_forward, mapeVal_forward, rsquareVal_forward] = evaluate_RELM(C_forward, H_train_numeric, Y_train, input_weights, bias, X_val, Y_val);

        % Debug print for forward values
        disp(['Forward: C = ', num2str(C_forward), ', RMSE = ', num2str(rmseVal_forward), ', MAE = ', num2str(maeVal_forward)]);
        epsilon = 1e-4; % atau nilai kecil lainnya yang sesuai
        % Periksa apakah metrik evaluasi lebih baik untuk maju
        if rmseVal_forward < best_rmse_val_forward && abs(best_rmse_val_forward - rmseVal_forward) > epsilon %|| (rmseVal_forward == best_rmse_val_forward && maeVal_forward < best_mae_val_forward)
            disp('Parameter 9');
            best_rmse_val_forward = rmseVal_forward;
            best_mae_val_forward = maeVal_forward;
            best_mape_val_forward = mapeVal_forward;
            best_rsquare_val_forward = rsquareVal_forward;
            best_C_forward = C_forward;
            no_change_forward = 0;
        else
            disp('Parameter 10');
            no_change_forward = no_change_forward + 1;
            if no_change_forward >= 5
                disp('Parameter 11');
                stop_forward = true;
            end
            disp(['No Change Forward: ', num2str(no_change_forward)]);
        end
        disp([' C: ', num2str(C_forward)]);
        disp('Validation Data');
        disp(['RMSE Val RELM: ', num2str(rmseVal_forward)]);
        disp(['MAE Val RELM: ', num2str(maeVal_forward)]);
        disp(['MAPE Val RELM: ', num2str(mapeVal_forward)]);
        disp(['R-Square Val RELM: ', num2str(rsquareVal_forward)]);
    end
    
    disp('Parameter 12');
    % Evaluasi model dengan nilai C mundur jika iterasi mundur belum dihentikan
    if ~stop_backward
        disp('Parameter 13');
        C_backward = C_backward / 10;
        [rmseVal_backward, maeVal_backward, mapeVal_backward, rsquareVal_backward] = evaluate_RELM(C_backward, H_train_numeric, Y_train, input_weights, bias, X_val, Y_val);

        % Debug print for backward values
        disp(['Backward: C = ', num2str(C_backward), ', RMSE = ', num2str(rmseVal_backward), ', MAE = ', num2str(maeVal_backward)]);

        % Periksa apakah metrik evaluasi lebih baik untuk mundur
        if rmseVal_backward < best_rmse_val_backward && abs(best_rmse_val_backward - rmseVal_backward) > epsilon %|| (rmseVal_backward == best_rmse_val_backward && maeVal_backward < best_mae_val_backward)
            disp('Parameter 14');
            best_rmse_val_backward = rmseVal_backward;
            best_mae_val_backward = maeVal_backward;
            best_mape_val_backward = mapeVal_backward;
            best_rsquare_val_backward = rsquareVal_backward;
            best_C_backward = C_backward;
            no_change_backward = 0;
        else
            disp('Parameter 15');
            no_change_backward = no_change_backward + 1;
            if no_change_backward >= 5
                disp('Parameter 16');
                stop_backward = true;
            end
            disp(['No Change backward: ', num2str(no_change_backward)]);
        end
        disp([' C: ', num2str(C_backward)]);
        disp('Validation Data');
        disp(['RMSE Val RELM: ', num2str(rmseVal_backward)]);
        disp(['MAE Val RELM: ', num2str(maeVal_backward)]);
        disp(['MAPE Val RELM: ', num2str(mapeVal_backward)]);
        disp(['R-Square Val RELM: ', num2str(rsquareVal_backward)]);
    end

    if best_rmse_val_forward < best_rmse_val || (best_rmse_val_forward == best_rmse_val && best_mae_val_forward < best_mae_val)
        best_rmse_val = best_rmse_val_forward;
        best_mae_val = best_mae_val_forward;
        best_mape_val = best_mape_val_forward;
        best_rsquare_val = best_rsquare_val_forward;
        best_C = best_C_forward;
    elseif best_rmse_val_backward < best_rmse_val || (best_rmse_val_backward == best_rmse_val && best_mae_val_backward < best_mae_val)
        best_rmse_val = best_rmse_val_backward;
        best_mae_val = best_mae_val_backward;
        best_mape_val = best_mape_val_backward;
        best_rsquare_val = best_rsquare_val_backward;
        best_C = best_C_backward;
    else
        disp('Tidak ada perubahan');
    end

    disp('Parameter 17');
    % Periksa kondisi berhenti
    if stop_forward && stop_backward
        disp('Parameter 18');
        break;
    end
    disp('Parameter 19');
end

disp('Setelah perulangan');
validationTime = toc;

% Test RELM on the testing set with best C
tic;
regularization_matrix = eye(hidden_layer_size) / best_C;
output_weights = pinv(H_train_numeric' * H_train_numeric + regularization_matrix) * (H_train_numeric' * Y_train);
H_test = max(0, input_weights * X_test' + repmat(bias, 1, size(X_test, 1)));
Y_OutTest = H_test' * output_weights;
executionTime = toc;

rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

disp('---------RELM----------');
disp(['Best C: ', num2str(best_C)]);
disp(' ');
disp('Validation Data');
disp(['RMSE Val RELM: ', num2str(best_rmse_val)]);
disp(['MAE Val RELM: ', num2str(best_mae_val)]);
disp(['MAPE Val RELM: ', num2str(best_mape_val)]);
disp(['R-Square Val RELM: ', num2str(best_rsquare_val)]);
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

% Fungsi untuk menghitung metrik evaluasi
function [rmseVal, maeVal, mapeVal, rsquareVal] = evaluate_RELM(C, H_train_numeric, Y_train, input_weights, bias, X_val, Y_val)
    regularization_matrix = eye(size(H_train_numeric, 2)) / C;
    output_weights = pinv(H_train_numeric' * H_train_numeric + regularization_matrix) * (H_train_numeric' * Y_train);
    H_val = max(0, input_weights * X_val' + repmat(bias, 1, size(X_val, 1)));
    Y_ValTest = H_val' * output_weights;
    rmseVal = rmse(Y_ValTest', Y_val');
    maeVal = mae(Y_ValTest', Y_val');
    mapeVal = mape(Y_ValTest', Y_val');
    rsquareVal = rsquare(Y_ValTest, Y_val);
end