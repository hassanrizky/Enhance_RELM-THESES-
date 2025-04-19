% Data preparation
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
% Number of hidden neurons
hidden_layer_size = 150;
input_size = size(X_train, 2);

% Randomly initialize input weights and bias using He initialization for ReLU
input_weights = randn(hidden_layer_size, input_size) * sqrt(2 / input_size);
bias = randn(hidden_layer_size, 1) * sqrt(2 / input_size);

% Calculate hidden layer output using ReLU activation
num_train = size(X_train, 1);
H_train = max(0, input_weights * X_train' + repmat(bias, 1, num_train));%relu

% Calculate output weights using Moore-Penrose generalized inverse
output_weights = pinv(H_train') * Y_train;
trainingTime = toc;

tic;
% Test ELM on the Validaton set
H_val = max(0, input_weights * X_val' + repmat(bias, 1, size(X_val,1))); %relu
Y_ValTest = H_val' * output_weights;
validationTime = toc;

rmseVal = rmse(Y_ValTest', Y_val');
maeVal = mae(Y_ValTest', Y_val');
mapeVal = mape(Y_ValTest', Y_val');
rsquareVal = rsquare(Y_ValTest, Y_val);

tic;
% Test ELM on the testing set
H_test = max(0, input_weights * X_test' + repmat(bias, 1, size(X_test, 1)));
Y_OutTest = H_test' * output_weights;
executionTime = toc;

% Evaluate performance
rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

disp('---------ELM----------');
disp('Validation Data');
disp(['RMSE Val ELM: ', num2str(rmseVal)]);
disp(['MAE Val ELM: ', num2str(maeVal)]);
disp(['MAPE Val ELM: ', num2str(mapeVal)]);
disp(['R-Square Val ELM: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test ELM: ', num2str(rmseTest)]);
disp(['MAE Test ELM: ', num2str(maeTest)]);
disp(['MAPE Test ELM: ', num2str(mapeTest)]);
disp(['R-Square Test ELM: ', num2str(rsquareTest)]);
disp(' ');
disp('Execution Time');
disp(['Training Time ELM: ', num2str(trainingTime)]);
disp(['Validation Time ELM: ', num2str(validationTime)]);
disp(['Testing Time ELM: ', num2str(executionTime)]);
disp(' ');

%%
% Validasi Silang (Cross-Validation)
num_folds = 5;

% Inisialisasi variabel untuk menyimpan hasil validasi silang
rmse_cv = zeros(num_folds, 1);
mae_cv = zeros(num_folds, 1);
mape_cv = zeros(num_folds, 1);
rsquare_cv = zeros(num_folds, 1);

rmseTest_cv = zeros(num_folds, 1);
maeTest_cv = zeros(num_folds, 1);
mapeTest_cv = zeros(num_folds, 1);
rsquareTest_cv = zeros(num_folds, 1);

executionTimes = zeros(1, num_folds);
tic;

% Lakukan cross-validation
cv = cvpartition(size(X_train, 1), 'KFold', num_folds);
for fold = 1:num_folds
    % Pisahkan data menjadi data pelatihan dan data validasi untuk lipatan ke-i
    X_train_cv = X_train(cv.training(fold), :);
    Y_train_cv = Y_train(cv.training(fold));
    X_val_cv = X_train(cv.test(fold), :);
    Y_val_cv = Y_train(cv.test(fold));
    
    % Inisialisasi bobot dan bias dengan inisialisasi He untuk relu
    input_weights = randn(best_hidden_layer_size, input_size) * sqrt(2 / input_size);
    bias = randn(best_hidden_layer_size, 1) * sqrt(2 / input_size);

    % Hitung hidden layer output dengan relu
    num_train_cv = size(X_train_cv, 1);
    H_train_cv = max(0, input_weights * X_train_cv' + repmat(bias, 1, num_train_cv));

    % Hitung output weights menggunakan Moore-Penrose generalized inverse
    output_weights_cv = pinv(H_train_cv') * Y_train_cv;

    % Hitung hidden layer output dan prediksi pada data validasi
    num_val_cv = size(X_val_cv, 1);
    H_val_cv = max(0, input_weights * X_val_cv' + repmat(bias, 1, num_val_cv));
    Y_OutVal_cv = H_val_cv' * output_weights_cv;

    % Hitung metrik kinerja untuk lipatan ke-i
    rmse_cv(fold) = rmse(Y_OutVal_cv', Y_val_cv');
    mae_cv(fold) = mae(Y_OutVal_cv', Y_val_cv');
    mape_cv(fold) = mape(Y_OutVal_cv', Y_val_cv');
    rsquare_cv(fold) = rsquare(Y_OutVal_cv, Y_val_cv);
    
    % Test ELM on the testing set
    H_test = max(0, input_weights * X_test' + repmat(bias, 1, size(X_test, 1)));
    Y_OutTest = H_test' * output_weights_cv;

    % Evaluate performance
    rmseTest_cv(fold) = rmse(Y_OutTest', Y_test');
    maeTest_cv(fold) = mae(Y_OutTest', Y_test');
    mapeTest_cv(fold) = mape(Y_OutTest', Y_test');
    rsquareTest_cv(fold) = rsquare(Y_OutTest, Y_test);

end
% Stop timing
executionTimes(fold) = toc;

% Hitung rata-rata metrik kinerja dari hasil cross-validation
avg_rmse_cv = mean(rmse_cv);
avg_mae_cv = mean(mae_cv);
avg_mape_cv = mean(mape_cv);
avg_rsquare_cv = mean(rsquare_cv);

avg_rmseTest_cv = mean(rmseTest_cv);
avg_maeTest_cv = mean(maeTest_cv);
avg_mapeTest_cv = mean(mapeTest_cv);
avg_rsquareTest_cv = mean(rsquareTest_cv);

% Tampilkan hasil cross-validation
disp('Hasil Cross-Validation:');
disp(['Rata-rata RMSE: ', num2str(avg_rmse_cv)]);
disp(['Rata-rata MAE: ', num2str(avg_mae_cv)]);
disp(['Rata-rata MAPE: ', num2str(avg_mape_cv)]);
disp(['Rata-rata R-squared: ', num2str(avg_rsquare_cv)]);
disp(['Average Execution Time: ', num2str(executionTimes), ' seconds']);
disp('  ');
disp('Testing Data');
disp(['RMSE Test ELM: ', num2str(avg_rmseTest_cv)]);
disp(['MAE Test ELM: ', num2str(avg_maeTest_cv)]);
disp(['MAPE Test ELM: ', num2str(avg_mapeTest_cv)]);
disp(['R-Square Test ELM: ', num2str(avg_rsquareTest_cv)]);
disp(' ');

%%
% Optimasi Hyperparameter Otomatis (Grid Search)
hidden_layer_sizes = [100, 500, 1000, 2000, 5000];  % Ganti dengan nilai hyperparameter yang sesuai

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

input_size = size(X_train, 2);
% Inisialisasi variabel untuk menyimpan hasil
best_rmse = Inf;
best_mae = Inf;
best_mape = Inf;
best_rsquare = Inf;
best_hidden_layer_size = 0;

% Lakukan Grid Search
for hidden_layer_size = hidden_layer_sizes
    % Inisialisasi bobot dan bias dengan inisialisasi He untuk relu
    input_weights = randn(hidden_layer_size, input_size) * sqrt(2 / input_size);
    bias = randn(hidden_layer_size, 1) * sqrt(2 / input_size);

    % Hitung hidden layer output dengan relu
    num_train = size(X_train, 1);
    H_train = max(0, input_weights * X_train' + repmat(bias, 1, num_train));

    % Hitung output weights menggunakan Moore-Penrose generalized inverse
    output_weights = pinv(H_train') * Y_train;

    % Hitung hidden layer output dan prediksi pada data validasi
    num_val = size(X_val, 1);
    H_val = max(0, input_weights * X_val' + repmat(bias, 1, num_val));
    Y_OutVal = H_val' * output_weights;

    % Hitung metrik kinerja untuk model dengan hyperparameter tertentu
    current_rmse = rmse(Y_OutVal', Y_val');
    current_mae = mae(Y_OutVal', Y_val');
    current_mape = mape(Y_OutVal', Y_val');
    current_rsquare = rsquare(Y_OutVal, Y_val);

    % Test ELM on the testing set
    H_test = max(0, input_weights * X_test' + repmat(bias, 1, size(X_test, 1)));
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
    disp(['RMSE Test ELM: ', num2str(rmseTest)]);
    disp(['MAE Test ELM: ', num2str(maeTest)]);
    disp(['MAPE Test ELM: ', num2str(mapeTest)]);
    disp(['R-Square Test ELM: ', num2str(rsquareTest)]);
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
