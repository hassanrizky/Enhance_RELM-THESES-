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

% Define polynomial degree
degree = 2; % Set the degree of the polynomial

% Fit Polynomial Regression model
X_poly_train = polyFeatures(X_train, degree);
modelPoly = fitrlinear(X_poly_train, Y_train);

% Validate the model
X_poly_val = polyFeatures(X_val, degree);
Y_ValTest = predict(modelPoly, X_poly_val);

rmseVal = rmse(Y_ValTest', Y_val');
maeVal = mae(Y_ValTest', Y_val');
mapeVal = mape(Y_ValTest', Y_val');
rsquareVal = rsquare(Y_ValTest, Y_val);

% Test the model
X_poly_test = polyFeatures(X_test, degree);
Y_OutTest = predict(modelPoly, X_poly_test);

rmseTest = rmse(Y_OutTest', Y_test');
maeTest = mae(Y_OutTest', Y_test');
mapeTest = mape(Y_OutTest', Y_test');
rsquareTest = rsquare(Y_OutTest, Y_test);

% Display results
disp('---------Polynomial Regression----------');
disp('Validation Data');
disp(['RMSE Val Polynomial: ', num2str(rmseVal)]);
disp(['MAE Val Polynomial: ', num2str(maeVal)]);
disp(['MAPE Val Polynomial: ', num2str(mapeVal)]);
disp(['R-Square Val Polynomial: ', num2str(rsquareVal)]);
disp(' ');
disp('Testing Data');
disp(['RMSE Test Polynomial: ', num2str(rmseTest)]);
disp(['MAE Test Polynomial: ', num2str(maeTest)]);
disp(['MAPE Test Polynomial: ', num2str(mapeTest)]);
disp(['R-Square Test Polynomial: ', num2str(rsquareTest)]);
disp(' ');

% Function to generate polynomial features
function X_poly = polyFeatures(X, degree)
    X_poly = ones(size(X, 1), 1);  % Initialize with intercept term
    
    for i = 1:degree
        X_poly = [X_poly X.^i];
    end
end