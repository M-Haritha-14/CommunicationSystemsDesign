clc;
close all;

tic
%load("data_file.mat");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Question 1(a)        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Defining layers for architecture-1
layers_1 = [
    sequenceInputLayer(2,"Name","Input_layer")
    fullyConnectedLayer(20,"Name","FullyConnected_layer_1")
    reluLayer("Name", "ReLU")
    fullyConnectedLayer(10,"Name","FullyConnected_layer_2")
    fullyConnectedLayer(2,"Name","FullyConnected_layer_3")
    tanhLayer("Name","tanh_layer")
    regressionLayer("Name", "Reg_layer")];
plot(layerGraph(layers_1));
set(findall(gcf,'-property','FontSize'),'FontSize',24)

 %% Training options
options_1 = trainingOptions('adam',...
'LearnRateSchedule', 'piecewise', ...
 'LearnRateDropPeriod', 1000, ...
 'ValidationData', {noisy_message_val, message_val_1}, ...
'MaxEpochs', 1000,...
'Plots', 'training-progress');


%% Training the network
net_1 = trainNetwork(noisy_message_train, message_train_1, layers_1, options_1);
toc

%% Testing the Trained Network

M = 4; % constellation order

constellation_symbols = M_QAM_constellation(M); % fetching 4-QAM symbols

Es_1 = ((norm(constellation_symbols))^2)/M; % average symbol energy

SNR_dB_1 = 0:1:10; %SNR in dB scale
    
SNR_1 = 10.^(SNR_dB_1/10); % SNR in linear scale

no_of_bits = 1e5; %samples

Tx_bits_1 = randi([0 1],[1,no_of_bits]); % random sequence of 1's 0's

Tx_crumb_decimal_1 = (bit2int(Tx_bits_1.',2))'; % converting each crumb to decimal

Tx_symbol_1 = bits2symbol_mapping(Tx_crumb_decimal_1,constellation_symbols); % mapping the bits to symbols

SNR_cnt_1 = numel(SNR_dB_1); % length of SNR

% AWGN noise vector
n_1 = randn([1,length(Tx_symbol_1)]) + 1i*randn([1,length(Tx_symbol_1)]);

BER_1 = zeros([1,SNR_cnt_1]);   %initializing BER array

for i = 1:1:SNR_cnt_1

    noise_1 = sqrt(Es_1/(2*SNR_1(i))).*n_1; %varying noise power as 1/SNR 

    Rx_symbol_1 = Tx_symbol_1 + noise_1; % tx symbols over AWGN

    ch_out_1 = Rx_symbol_1; % output of the channel

    channel_out_1 = [real(ch_out_1);imag(ch_out_1)]; % complex number as 2D vector
    
    hat_1 = sign(predict(net_1, channel_out_1)); % prediction of the network
    
    y_hat_1 = hat_1(1,:) + 1i*hat_1(2,:); % predicted symbol

    decoded_sym_1 =  ML_decoder(y_hat_1,constellation_symbols); %decoded symbols

    decoded_bits_1 = symbol2bits_mapping(decoded_sym_1,constellation_symbols);%decoded bits sequence

    bits_in_error_1 = nnz(decoded_bits_1 ~= Tx_bits_1); % no of received bits in error 

    BER_1(i) = bits_in_error_1/no_of_bits; % BER 
end

%% Plot results
figure();
semilogy(0:1:10, BER_1, 'LineWidth', 2, 'Marker', '+', 'MarkerSize', 10);
grid on; hold on;
semilogy(0:1:10, QAM_4_ML_BER, 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 10)
legend("Neural detector - 1", "ML detector");
title("Setup 1")
xlabel("SNR (in dB)")
ylabel("BER")
set(findall(gcf,'-property','FontSize'),'FontSize',24)

%% saving the model parameters
% model_output_1 = net;
% save model_output_1