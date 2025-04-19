X = data(:, 2:100);
Y = data(:, 101);

rng(82)
index00 = randperm(1000);

trainIndex = index00(1:600);
validationIndex = index00(601:700);
testIndex = index00(701:1000);

X_train = X(trainIndex, :);
sample_rtt = X(1, :);
%sample_rtt = [100, 110, 105, 120, 115, 125];
% Parameter
alpha = 0.125;
beta = 0.25;


% Inisialisasi EstimatedRTT dan RTTVar
estimated_rtt = sample_rtt(1);  % Anggap nilai awal adalah sampel pertama
rttvar = 0;

% Loop melalui sample RTT untuk memperbarui EstimatedRTT dan RTTVar
for i = 2:length(sample_rtt)
    rtt_sample = sample_rtt(i);
    
    % Hitung RTTVar
    rttvar = (1 - beta) * rttvar + beta * abs(rtt_sample - estimated_rtt);
    
    % Hitung EstimatedRTT
    estimated_rtt = (1 - alpha) * estimated_rtt + alpha * rtt_sample;
    
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

%[f_rtt, x_rtt] = ecdf(sample_rtt);
%f_est, x_est] = ecdf(estimated_rtt);
%plot(x_rtt, f_rtt, 'b', x_est, f_est, 'r');
%legend('Measured RTT', 'Estimated RTT');