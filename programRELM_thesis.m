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
hidden_layer_size = 150;
C = 0.00000001;

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

%%
% Plot the original data and the BiLSTM predictions for the first 100 points
figure;
plot(1:length(Y_test), Y_test, 'o-', 'DisplayName', 'Original Data');
hold on;
plot(1:length(Y_OutTest), Y_OutTest, 'x-', 'DisplayName', 'RELM Predictions');
xlabel('Delay Value');
ylabel('RTT Value');
title('Round Trip TIme Prediction using RELM');
legend
hold off;
%% Menggunakan L1 Regularization (L1-Norm)
% Load data
X = data(:, 2:101);
Y = data(:, 1);

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

% Set parameters for RELM
hidden_layer_size = 100;
C = 0.00000000001;  % Regularization parameter (small value for Lasso effect)

% Initialize input weights and bias
input_size = size(X_train, 2);
input_weights = randn(hidden_layer_size, input_size) * sqrt(2 / input_size);
bias = randn(hidden_layer_size, 1) * sqrt(2 / input_size);

% Calculate hidden layer output (ReLU activation)
H_train = max(0, input_weights * X_train' + repmat(bias, 1, size(X_train, 1)));

% Regularization matrix for L1 (Lasso)
%regularization_matrix = eye(hidden_layer_size) * C;

% Calculate output weights using Lasso Regression
H_train_numeric = double(H_train');
output_weights = lasso(H_train_numeric, Y_train, 'Lambda', C);
%output_weights = output_weights';

% Test RELM on the validation set
H_val = max(0, input_weights * X_val' + repmat(bias, 1, size(X_val, 1)));
Y_ValTest = H_val' * output_weights;

% Calculate performance metrics for validation set
rmseVal = rmse(Y_ValTest', Y_val');
maeVal = mae(Y_ValTest', Y_val');
mapeVal = mape(Y_ValTest', Y_val');
rsquareVal = rsquare(Y_ValTest, Y_val);

% Test RELM on the testing set
H_test = max(0, input_weights * X_test' + repmat(bias, 1, size(X_test, 1)));
Y_OutTest = H_test' * output_weights;

% Calculate performance metrics for testing set
rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

% Display results
disp('---------RELM with Lasso Regression----------');
disp('Validation Data');
disp(['RMSE Val RELM Lasso: ', num2str(rmseVal)]);
disp(['MAE Val RELM Lasso: ', num2str(maeVal)]);
disp(['MAPE Val RELM Lasso: ', num2str(mapeVal)]);
disp(['R-Square Val RELM Lasso: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test RELM Lasso: ', num2str(rmseTest)]);
disp(['MAE Test RELM Lasso: ', num2str(maeTest)]);
disp(['MAPE Test RELM Lasso: ', num2str(mapeTest)]);
disp(['R-Square Test RELM Lasso: ', num2str(rsquareTest)]);
%% Menggunakan L2 Regularization (Ridge Regression)
% Load data
X = data(:, 2:101);
Y = data(:, 1);

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

% Set parameters for RELM
hidden_layer_size = 100;
C = 0.00000000001;  % Regularization parameter (small value for Ridge effect)

% Initialize input weights and bias
input_size = size(X_train, 2);
input_weights = randn(hidden_layer_size, input_size) * sqrt(2 / input_size);
bias = randn(hidden_layer_size, 1) * sqrt(2 / input_size);

% Calculate hidden layer output (ReLU activation)
H_train = max(0, input_weights * X_train' + repmat(bias, 1, size(X_train, 1)));

% Regularization matrix for L2 (Ridge)
regularization_matrix = eye(hidden_layer_size) * C;

% Calculate output weights using Ridge Regression
H_train_numeric = double(H_train');
output_weights = pinv(H_train_numeric' * H_train_numeric + regularization_matrix) * (H_train_numeric' * Y_train);

% Test RELM on the validation set
H_val = max(0, input_weights * X_val' + repmat(bias, 1, size(X_val, 1)));
Y_ValTest = H_val' * output_weights;

% Calculate performance metrics for validation set
rmseVal = rmse(Y_ValTest', Y_val');
maeVal = mae(Y_ValTest', Y_val');
mapeVal = mape(Y_ValTest', Y_val');
rsquareVal = rsquare(Y_ValTest, Y_val);

% Test RELM on the testing set
H_test = max(0, input_weights * X_test' + repmat(bias, 1, size(X_test, 1)));
Y_OutTest = H_test' * output_weights;

% Calculate performance metrics for testing set
rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

% Display results
disp('---------RELM with Ridge Regression----------');
disp('Validation Data');
disp(['RMSE Val RELM Ridge: ', num2str(rmseVal)]);
disp(['MAE Val RELM Ridge: ', num2str(maeVal)]);
disp(['MAPE Val RELM Ridge: ', num2str(mapeVal)]);
disp(['R-Square Val RELM Ridge: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test RELM Ridge: ', num2str(rmseTest)]);
disp(['MAE Test RELM Ridge: ', num2str(maeTest)]);
disp(['MAPE Test RELM Ridge: ', num2str(mapeTest)]);
disp(['R-Square Test RELM Ridge: ', num2str(rsquareTest)]);

%%
% Validasi Silang (Cross-Validation)
num_folds = 5;

X = data(:, 2:100);
Y = data(:, 101);

rng(82)
index00 = randperm(1000);

trainIndex = index00(1:1000);
%validationIndex = index00(601:700);
%testIndex = index00(701:1000);

X_train = X(trainIndex, :);
Y_train = Y(trainIndex, 1);

%X_val = X(validationIndex, :);
%Y_val = Y(validationIndex, 1);

%X_test = X(testIndex, :);
%Y_test = Y(testIndex, 1);

tic;
%MODELRELM
% Set parameters for RELM
hidden_layer_size = 150;
C = 0.00000001;


% Inisialisasi variabel untuk menyimpan hasil validasi silang
rmse_cv = zeros(num_folds, 1);
mae_cv = zeros(num_folds, 1);
mape_cv = zeros(num_folds, 1);
rsquare_cv = zeros(num_folds, 1);
executionTimes_cv = zeros(num_folds, 1);


% Lakukan cross-validation
cv = cvpartition(size(X_train, 1), 'KFold', num_folds);
for fold = 1:num_folds
    % Pisahkan data menjadi data pelatihan dan data validasi untuk lipatan ke-i
    X_train_cv = X_train(cv.training(fold), :);
    Y_train_cv = Y_train(cv.training(fold));
    X_val_cv = X_train(cv.test(fold), :);
    Y_val_cv = Y_train(cv.test(fold));
    tic;

    % Inisialisasi bobot dan bias dengan inisialisasi He untuk relu
    input_size = size(X_train_cv, 2);
    input_weights = randn(hidden_layer_size, input_size) * sqrt(2 / input_size);
    bias = randn(hidden_layer_size, 1) * sqrt(2 / input_size);

    % Hitung hidden layer output dengan relu
    num_train_cv = size(X_train_cv, 1);
    %H_train_cv = tanh(input_weights*X_train_cv' + repmat(bias, 1, num_train_cv)); %tanh
    H_train_cv = max(0, input_weights * X_train_cv' + repmat(bias, 1, num_train_cv)); %relu
    %H_train_cv = 1 ./ (1 + exp(-(input_weights * X_train_cv' + repmat(bias, 1, num_train_cv)))); % sigmoid

    % Regularization matrix
    regularization_matrix_cv = eye(hidden_layer_size) / C;

    % Hitung output weights menggunakan Moore-Penrose generalized inverse
    H_train_numeric_cv = double(H_train_cv');
    output_weights_cv = pinv(H_train_numeric_cv' * H_train_numeric_cv + regularization_matrix_cv) * (H_train_numeric_cv' * Y_train_cv);

    % Hitung hidden layer output dan prediksi pada data validasi
    num_val_cv = size(X_val_cv, 1);
    %H_val_cv = tanh(input_weights*X_val_cv' + repmat(bias, 1, num_val_cv)))); %tanh
    H_val_cv = max(0, input_weights * X_val_cv' + repmat(bias, 1, num_val_cv)); %relu
    %H_val_cv = 1 ./ (1 + exp(-(input_weights * X_val_cv' + repmat(bias, 1, num_val_cv)))); %sigmoid
    Y_OutVal_cv = H_val_cv' * output_weights_cv;
    executionTimes_cv(fold) = toc;

    % Hitung metrik kinerja untuk lipatan ke-i
    rmse_cv(fold) = rmse(Y_OutVal_cv', Y_val_cv');
    mae_cv(fold) = mae(Y_OutVal_cv', Y_val_cv');
    mape_cv(fold) = mape(Y_OutVal_cv', Y_val_cv');
    rsquare_cv(fold) = rsquare(Y_OutVal_cv, Y_val_cv);

end
% Stop timing

% Hitung rata-rata metrik kinerja dari hasil cross-validation
avg_rmse_cv = mean(rmse_cv);
avg_mae_cv = mean(mae_cv);
avg_mape_cv = mean(mape_cv);
avg_rsquare_cv = mean(rsquare_cv);
avg_executionTimes_cv = mean(executionTimes_cv);

% Tampilkan hasil cross-validation
disp('Hasil Cross-Validation:');
disp(['Rata-rata RMSE: ', num2str(avg_rmse_cv)]);
disp(['Rata-rata MAE: ', num2str(avg_mae_cv)]);
disp(['Rata-rata MAPE: ', num2str(avg_mape_cv)]);
disp(['Rata-rata R-squared: ', num2str(avg_rsquare_cv)]);
disp(['Average Execution Time: ', num2str(avg_executionTimes_cv), ' seconds']);
disp('  ');

%%
% Optimasi Hyperparameter Otomatis (Grid Search)
hidden_layer_sizes = [50, 100, 150, 500, 1000, 5000];  % Ganti dengan nilai hyperparameter yang sesuai

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

C = 1;
input_size = size(X_train, 2);
% Inisialisasi variabel untuk menyimpan hasil
best_rmse = Inf;
best_mae = Inf;
best_mape = Inf;
best_rsquare = Inf;
best_hidden_layer_size = 0;

% Lakukan Grid Search
for hidden_layer_size = hidden_layer_sizes
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
    regularization_matrix = eye(hidden_layer_size) / C;

    % Hitung output weights menggunakan Moore-Penrose generalized inverse
    H_train_numeric = double(H_train');
    output_weights = pinv(H_train_numeric' * H_train_numeric + regularization_matrix) * (H_train_numeric' * Y_train);

    % Hitung hidden layer output dan prediksi pada data validasi
    % Test RELM on the Validaton set
    %H_val = tanh(input_weights*X_val' + repmat(bias, 1, size(X_val, 1))); %tanh
    H_val = max(0, input_weights * X_val' + repmat(bias, 1, size(X_val,1))); %relu
    %H_val = 1 ./ (1 + exp(-(input_weights * X_val' + repmat(bias, 1, size(X_val, 1))))); %sigmoid
    Y_OutVal = H_val' * output_weights;

    % Hitung metrik kinerja untuk model dengan hyperparameter tertentu
    current_rmse = rmse(Y_OutVal', Y_val');
    current_mae = mae(Y_OutVal', Y_val');
    current_mape = mape(Y_OutVal', Y_val');
    current_rsquare = rsquare(Y_OutVal, Y_val);
    
    % Test ELM on the testing set
    %H_test = tanh(input_weights*X_test' + repmat(bias, 1, size(X_test, 1))); %tanh
    H_test = max(0, input_weights * X_test' + repmat(bias, 1, size(X_test,1))); %relu
    %H_test = 1 ./ (1 + exp(-(input_weights * X_test' + repmat(bias, 1, size(X_test, 1))))); %sigmoid
    Y_OutTest = H_test' * output_weights;

    % Evaluate performance
    rmseTest = rmse(Y_OutTest', Y_test');
    maeTest = mae(Y_OutTest', Y_test');
    mapeTest = mape(Y_OutTest', Y_test');
    rsquareTest = rsquare(Y_OutTest, Y_test);

    % Perbarui hyperparameter terbaik jika ditemukan hasil yang lebih baik
    if current_rmse < best_rmse
        best_input_weight = input_weights;
        best_bias = bias;
        best_output_weights = output_weights;
        best_rmse = current_rmse;
        best_mae = current_mae;
        best_mape = current_mape;
        best_rsquare = current_rsquare;
        best_hidden_layer_size = hidden_layer_size;
    end

    disp(['Ukuran Hidden Layer : ', num2str(hidden_layer_size)]);
    disp('---------------------');
    disp(['RMSE Terbaik: ', num2str(current_rmse)]);
    disp(['MAE Terbaik: ', num2str(current_mae)]);
    disp(['MAPE Terbaik: ', num2str(current_mape)]);
    disp(['R-Square Terbaik: ', num2str(current_rsquare)]);
    disp('  ');
    disp('Testing Data');
    disp(['RMSE Test RELM: ', num2str(rmseTest)]);
    disp(['MAE Test RELM: ', num2str(maeTest)]);
    disp(['MAPE Test RELM: ', num2str(mapeTest)]);
    disp(['R-Square Test RELM: ', num2str(rsquareTest)]);
    disp(' ');
end

% Tampilkan hasil hyperparameter terbaik
disp('---------------------');
disp('---------------------');
disp('Hyperparameter Terbaik:');
disp(['Ukuran Hidden Layer Terbaik: ', num2str(best_hidden_layer_size)]);
disp('---------------------');
disp(['RMSE Terbaik: ', num2str(best_rmse)]);
disp(['MAE Terbaik: ', num2str(best_mae)]);
disp(['MAPE Terbaik: ', num2str(best_mape)]);
disp(['R-Square Terbaik: ', num2str(best_rsquare)]);

%%
H_test = max(0, best_input_weight * X_test' + repmat(best_bias, 1, size(X_test, 1))); %relu
Y_OutTest = H_test' * best_output_weights;

rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

disp(['RMSE Data Test: ', num2str(rmseTest)]);
disp(['MAE Data Test: ', num2str(maeTest)]);
disp(['MAPE Data Test: ', num2str(mapeTest)]);
disp(['R-Square Data Test: ', num2str(rsquareTest)]);

%%

% Plot hasil prediksi terhadap nilai sebenarnya pada data validasi
figure;
scatter(Y_val', Y_ValTest');
hold on;
plot([min(Y_val'), max(Y_val')], [min(Y_val'), max(Y_val')], '-r'); % Garis referensi untuk prediksi yang sempurna
xlabel('True Values');
ylabel('Predicted Values');
title('Validation Set: True vs Predicted Values');
legend('Data Points', 'Perfect Prediction Line', 'Location', 'Best');
grid on;
hold off;


%%
% Plot hasil prediksi terhadap nilai sebenarnya pada data validasi
figure;
scatter(Y_test', Y_OutTest');
hold on;
plot([min(Y_test'), max(Y_test')], [min(Y_test'), max(Y_test')], '-r'); % Garis referensi untuk prediksi yang sempurna
xlabel('True Values');
ylabel('Predicted Values');
title('Validation Set: True vs Predicted Values');
legend('Data Points', 'Perfect Prediction Line', 'Location', 'Best');
grid on;
hold off;

%%

% Membuat model neural network
hidden_layer_size_nn = 100;
net_nn = fitnet(hidden_layer_size_nn);

% Melatih model pada data pelatihan
net_nn.trainParam.epochs = 100;
net_nn = train(net_nn, X_train', Y_train');

% Melakukan prediksi pada data validasi
Y_ValTest_NN = net_nn(X_val');

% Evaluasi pada data validasi
rmseVal_NN = rmse(Y_ValTest_NN, Y_val');
maeVal_NN = mae(Y_ValTest_NN, Y_val');
mapeVal_NN = mape(Y_ValTest_NN, Y_val');
rsquareVal_NN = rsquare(Y_ValTest_NN, Y_val);

% Melakukan prediksi pada data pengujian
Y_OutTest_NN = net_nn(X_test');

% Evaluasi pada data pengujian
rmseTest_NN = rmse(Y_OutTest_NN, Y_test');
maeTest_NN = mae(Y_OutTest_NN, Y_test');
mapeTest_NN = mape(Y_OutTest_NN, Y_test');
rsquareTest_NN = rsquare(Y_OutTest_NN, Y_test);

disp('---------Neural Network----------');
disp('Validation Data');
disp(['RMSE Val NN: ', num2str(rmseVal_NN)]);
disp(['MAE Val NN: ', num2str(maeVal_NN)]);
disp(['MAPE Val NN: ', num2str(mapeVal_NN)]);
disp(['R-Square Val NN: ', num2str(rsquareVal_NN)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test NN: ', num2str(rmseTest_NN)]);
disp(['MAE Test NN: ', num2str(maeTest_NN)]);
disp(['MAPE Test NN: ', num2str(mapeTest_NN)]);
disp(['R-Square Test NN: ', num2str(rsquareTest_NN)]);
disp(' ');

%%

X = data(:, 2:101);
Y = data(:, 1);

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
C = 0.000000000001;

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
alpha = 100;  % You can adjust the alpha parameter as needed
%H_train = tanh(input_weights*X_train' + repmat(bias, 1, num_train)); %tanh
H_train = max(alpha * X_train' + repmat(bias, 1, num_train), input_weights * X_train' + repmat(bias, 1, num_train)); %relu
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
H_val = max(alpha * X_val' + repmat(bias, 1, size(X_val, 1)), input_weights * X_val'); %relu
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
H_test = max(alpha * X_test' + repmat(bias, 1, size(X_test, 1)), input_weights * X_test'); %relu
%H_test = 1 ./ (1 + exp(-(input_weights * X_test' + repmat(bias, 1, size(X_test, 1))))); %sigmoid
Y_OutTest = H_test' * output_weights;
executionTime = toc;

rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

disp('---------RELM (Leaky ReLU)----------');
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

%%
% Algoritma untuk nyari nilai C di RELM (menggunakan Brute Force)

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
hidden_layer_size = 150;

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
trainingTime = toc;

tic;
best_C = 0;
best_rmse_val = Inf;
best_mae_val = Inf;
best_mape_val = Inf;
best_rsquare_val = -Inf;
H_train_numeric = double(H_train');

% Rentang nilai C yang akan diuji
C_range = 10 .^ (-12:1:12); % Membuat rentang nilai C dari 1e12 hingga 1e-12
% Inisialisasi array untuk menyimpan nilai RMSE dan MAE
rmse_values = zeros(size(C_range));
mae_values = zeros(size(C_range));

for i = 1:length(C_range)
    % Train RELM
    regularization_matrix = eye(hidden_layer_size)/C_range(i);
    output_weights = pinv(H_train_numeric' * H_train_numeric + regularization_matrix) * (H_train_numeric' *Y_train);
    
    % Test RELM on the Validation set
    H_val = max(0, input_weights * X_val' + repmat(bias, 1, size(X_val,1)));
    Y_ValTest = H_val' * output_weights;
    
    % Hitung metrik evaluasi
    rmseVal = rmse(Y_ValTest', Y_val');
    rmse_values(i) = rmseVal;

    maeVal = mae(Y_ValTest', Y_val');
    mae_values(i) = maeVal;

    mapeVal = mape(Y_ValTest', Y_val');
    rsquareVal = rsquare(Y_ValTest, Y_val);
    
    disp([' C: ', num2str(C_range(i))]);
    disp('Validation Data');
    disp(['RMSE Val RELM: ', num2str(rmseVal)]);
    disp(['MAE Val RELM: ', num2str(maeVal)]);
    disp(['MAPE Val RELM: ', num2str(mapeVal)]);
    disp(['R-Square Val RELM: ', num2str(rsquareVal)]);
    
    % Memperbarui nilai C terbaik jika ditemukan nilai evaluasi yang lebih baik
    if rmseVal < best_rmse_val || (rmseVal == best_rmse_val && maeVal < best_mae_val)
        best_rmse_val = rmseVal;
        best_C = C_range(i);
        best_mae_val = maeVal;
        best_mape_val = mapeVal;
        best_rsquare_val = rsquareVal;
    end
end

validationTime = toc;
% Regularization matrix
%regularization_matrix = eye(hidden_layer_size)/C;

% Calculate output weights using Moore-Penrose generalized inverse
%H_train_numeric = double(H_train');
%output_weights = pinv(H_train_numeric' * H_train_numeric + regularization_matrix) * (H_train_numeric' *Y_train);
%trainingTime = toc;

%tic;
% Test RELM on the Validaton set
%H_val = tanh(input_weights*X_val' + repmat(bias, 1, size(X_val, 1))); %tanh
%H_val = max(0, input_weights * X_val' + repmat(bias, 1, size(X_val,1))); %relu
%H_val = 1 ./ (1 + exp(-(input_weights * X_val' + repmat(bias, 1, size(X_val, 1))))); %sigmoid
%Y_ValTest = H_val' * output_weights;
%validationTime = toc;

%rmseVal = rmse(Y_ValTest', Y_val');
%maeVal = mae(Y_ValTest', Y_val');
%mapeVal = mape(Y_ValTest', Y_val');
%rsquareVal = rsquare(Y_ValTest, Y_val);

tic;
% Test RELM on the testing set
regularization_matrix = eye(hidden_layer_size)/best_C;
output_weights = pinv(H_train_numeric' * H_train_numeric + regularization_matrix) * (H_train_numeric' *Y_train);
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

% Plot grafik RMSE dan MAE
figure;
semilogx(C_range, rmse_values, 'b', 'LineWidth', 2);
xlabel('C Value (log scale)');
ylabel('RMSE');
title('RMSE Graph from 10^{-12} to 10^{12}');
grid on;

figure;
semilogx(C_range, mae_values, 'b', 'LineWidth', 2);
xlabel('C Value (log scale)');
ylabel('MAE');
title('MAE Graph from 10^{-12} to 10^{12}');
grid on;

%% Split data tanpa diacak

% Assuming 'data' is already loaded and structured as in your original code

X = data(:, 2:101);
Y = data(:, 1);

% Split the data without randomization
trainIndex = 1:600;
validationIndex = 601:700;
testIndex = 701:1000;

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

%% Algoritma proposed untuk nilai C (Masih error)

X = data(:, 2:101);
Y = data(:, 1);

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

% Train RELM (DASAR YANG DIGUNAKAN)
input_size = size(X_train, 2);

% Inisialisasi bobot dan bias dengan inisialisasi He untuk relU
input_weights = randn(hidden_layer_size, input_size) * sqrt(2 / input_size);
bias = randn(hidden_layer_size, 1) * sqrt(2 / input_size);

% Calculate hidden layer output
num_train = size(X_train, 1);
H_train = max(0, input_weights * X_train' + repmat(bias, 1, num_train)); %relu
H_train_numeric = double(H_train');
trainingTime = toc;

% Inisialisasi variabel untuk pencarian nilai C
C = 1;
best_C = C;
best_rmse_val = Inf;
best_mae_val = Inf;
best_mape_val = Inf;
best_rsquare_val = -Inf;

% Variabel untuk menghitung perulangan
n_forward = 0;
n_backward = 0;
no_change_forward = 0;
no_change_backward = 0;
disp('sebelum perulangan');
while true
    disp('bagian 1');
    % Train RELM
    regularization_matrix = eye(hidden_layer_size) / C;
    output_weights = pinv(H_train_numeric' * H_train_numeric + regularization_matrix) * (H_train_numeric' * Y_train);
    
    % Test RELM on the Validation set
    H_val = max(0, input_weights * X_val' + repmat(bias, 1, size(X_val, 1)));
    Y_ValTest = H_val' * output_weights;
    
    % Hitung metrik evaluasi
    rmseVal = rmse(Y_ValTest', Y_val');
    maeVal = mae(Y_ValTest', Y_val');
    mapeVal = mape(Y_ValTest', Y_val');
    rsquareVal = rsquare(Y_ValTest, Y_val);
    
    if rmseVal < best_rmse_val || (rmseVal == best_rmse_val && maeVal < best_mae_val)
        best_rmse_val = rmseVal;
        best_mae_val = maeVal;
        best_mape_val = mapeVal;
        best_rsquare_val = rsquareVal;
        best_C = C;
        disp('bagian 2');
    else
        disp('bagian 3');
        if C > 1
            no_change_forward = no_change_forward + 1;
        else
            no_change_backward = no_change_backward + 1;
        end
    end
    
    if no_change_forward >= 5 && no_change_backward >= 5
        disp('bagian 4');
        break;
    end
    
    disp('bagian 5');
    % Update nilai C
    if no_change_forward < 5
        disp('bagian 6');
        C = C * 10; % Maju
        n_forward = n_forward + 1;
    elseif no_change_backward < 5
        disp('bagian 7');
        C = C / 10; % Mundur
        n_backward = n_backward + 1;
    end
    
    % Jika performa dengan nilai C yang baru lebih buruk, break loop
    if rmseVal > best_rmse_val
        disp('bagian 8');
        break;
    end
end

validationTime = toc;
disp('diluar perulangan');
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

% Plot grafik RMSE dan MAE
%figure;
%semilogx(C_range, rmse_values, 'b', 'LineWidth', 2);
%xlabel('Nilai C (log scale)');
%ylabel('RMSE');
%title('Grafik RMSE dari 10^{-12} sampai 10^{12}');
%grid on;

%figure;
%semilogx(C_range, mae_values, 'b', 'LineWidth', 2);
%xlabel('Nilai C (log scale)');
%ylabel('RMSE');
%title('Grafik MAE dari 10^{-12} sampai 10^{12}');
%grid on;

%% Algoritma proposed untuk nilai C part 2

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
% MODELRELM
% Set parameters for RELM
hidden_layer_size = 150;

% Train RELM (DASAR YANG DIGUNAKAN)
input_size = size(X_train, 2);

% Inisialisasi bobot dan bias dengan inisialisasi He untuk relU
input_weights = randn(hidden_layer_size, input_size) * sqrt(2 / input_size);
bias = randn(hidden_layer_size, 1) * sqrt(2 / input_size);

% Calculate hidden layer output
num_train = size(X_train, 1);
H_train = max(0, input_weights * X_train' + repmat(bias, 1, num_train)); % relu
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

initial_iteration = true;  % Variabel untuk menandai iterasi pertama
initial_rmse = inf;  % Untuk menyimpan nilai RMSE awal
initial_mae = inf;   % Untuk menyimpan nilai MAE awal
initial_mape = inf;  % Untuk menyimpan nilai MAPE awal
initial_rsquare = -inf; % Untuk menyimpan nilai R-squared awal
while true
    % Evaluasi model dengan nilai C saat ini, hanya jika belum ada perubahan
    if initial_iteration
        [rmseVal, maeVal, mapeVal, rsquareVal] = evaluate_RELM(C, H_train_numeric, Y_train, input_weights, bias, X_val, Y_val);
        if initial_rmse == inf
            % Simpan nilai awal saat C = 1
            initial_rmse = rmseVal;
            initial_mae = maeVal;
            initial_mape = mapeVal;
            initial_rsquare = rsquareVal;
        end
        if rmseVal < best_rmse_val || (rmseVal == best_rmse_val && maeVal < best_mae_val);
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
    
    % Evaluasi model dengan nilai C maju jika iterasi maju belum dihentikan
    if ~stop_forward
        C_forward = C_forward * 10;
        [rmseVal_forward, maeVal_forward, mapeVal_forward, rsquareVal_forward] = evaluate_RELM(C_forward, H_train_numeric, Y_train, input_weights, bias, X_val, Y_val);
 
        epsilon = 1e-4; % atau nilai kecil lainnya yang sesuai
        % Periksa apakah metrik evaluasi lebih baik untuk maju
        if rmseVal_forward < best_rmse_val_forward && abs(best_rmse_val_forward - rmseVal_forward) > epsilon %|| (rmseVal_forward == best_rmse_val_forward && maeVal_forward < best_mae_val_forward)
            best_rmse_val_forward = rmseVal_forward;
            best_mae_val_forward = maeVal_forward;
            best_mape_val_forward = mapeVal_forward;
            best_rsquare_val_forward = rsquareVal_forward;
            best_C_forward = C_forward;
            no_change_forward = 0;
        else
            no_change_forward = no_change_forward + 1;
            if no_change_forward >= 5
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
    % Evaluasi model dengan nilai C mundur jika iterasi mundur belum dihentikan
    if ~stop_backward
        C_backward = C_backward / 10;
        [rmseVal_backward, maeVal_backward, mapeVal_backward, rsquareVal_backward] = evaluate_RELM(C_backward, H_train_numeric, Y_train, input_weights, bias, X_val, Y_val);

        % Periksa apakah metrik evaluasi lebih baik untuk mundur
        if rmseVal_backward < best_rmse_val_backward && abs(best_rmse_val_backward - rmseVal_backward) > epsilon %|| (rmseVal_backward == best_rmse_val_backward && maeVal_backward < best_mae_val_backward)
            best_rmse_val_backward = rmseVal_backward;
            best_mae_val_backward = maeVal_backward;
            best_mape_val_backward = mapeVal_backward;
            best_rsquare_val_backward = rsquareVal_backward;
            best_C_backward = C_backward;
            no_change_backward = 0;
        else
            no_change_backward = no_change_backward + 1;
            if no_change_backward >= 5
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

    % Periksa kondisi berhenti
    if stop_forward && stop_backward
        break;
    end
end
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
% Plot grafik RMSE dan MAE
% figure;
% semilogx(C_range, rmse_values, 'b', 'LineWidth', 2);
% xlabel('Nilai C (log scale)');
% ylabel('RMSE');
% title('Grafik RMSE dari 10^{-12} sampai 10^{12}');
% grid on;

% figure;
% semilogx(C_range, mae_values, 'b', 'LineWidth', 2);
% xlabel('Nilai C (log scale)');
% ylabel('RMSE');
% title('Grafik MAE dari 10^{-12} sampai 10^{12}');
% grid on;

% disp('Nilai C yang diuji:');
% disp(C_values);

%%