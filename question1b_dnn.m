clc;
close all;

tic
%load("data_file.mat");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Question 1(b)        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Defining layers for architecture-2
layers_2 = [
    sequenceInputLayer(2,"Name","Input_layer")
    fullyConnectedLayer(20,"Name","FullyConnected_layer_1")
    reluLayer("Name", "ReLU")
    fullyConnectedLayer(10,"Name","FullyConnected_layer_2")
    fullyConnectedLayer(4,"Name","FullyConnected_layer_3")
    softmaxLayer("Name","Softmax_layer")
    classificationLayer("Name", "Classification_layer")];
plot(layerGraph(layers_2));
set(findall(gcf,'-property','FontSize'),'FontSize',24)

 %% Training options
options_2 = trainingOptions('adam',...
'LearnRateSchedule', 'piecewise', ...
 'LearnRateDropPeriod', 1000, ...
 'MaxEpochs', 1000,...
 'ValidationData', {noisy_message_val, message_val_2}, ...
'Plots', 'training-progress');

%% Training the network
net_2 = trainNetwork(noisy_message_train, message_train_2,layers_2, options_2);
toc

%% Testing the Trained Network
M = 4; % constellation order

constellation_symbols = M_QAM_constellation(M); % fetching 4-QAM symbols

Es = ((norm(constellation_symbols))^2)/M; % average symbol energy

SNR_dB = 0:1:10; %SNR in dB scale
    
SNR = 10.^(SNR_dB/10); % SNR in linear scale

no_of_bits = 1e6; %samples

Tx_bits_2 = randi([0 1],[1,no_of_bits]); % random sequence of 1's 0's

Tx_crumb_decimal_2 = (bit2int(Tx_bits_2.',2))'; % converting each nibble to decimal

Tx_symbol_2 = bits2symbol_mapping(Tx_crumb_decimal_2,constellation_symbols); % mapping the bits to symbols

SNR_cnt_2 = numel(SNR_dB); % length of SNR

% AWGN noise vector
n_2 = randn([1,length(Tx_symbol_2)]) + 1i*randn([1,length(Tx_symbol_2)]);

BER_2 = zeros([1,SNR_cnt_2]);   %initializing BER array

for i = 1:1:SNR_cnt_2

    noise_2 = sqrt(Es/(2*SNR(i))).*n_2; %varying noise power as 1/SNR since Es = 1

    Rx_symbol_2 = Tx_symbol_2 + noise_2; % tx symbols over AWGN

    ch_out_2 = Rx_symbol_2;

    channel_out_2 = [real(ch_out_2);imag(ch_out_2)]; % complex output as 2D vector
    
    hat_2 = predict(net_2, channel_out_2); % 
    
    [~,idx] = max(hat_2);

    y_hat_2 = constellation_symbols(idx);

    decoded_sym_2 =  ML_decoder(y_hat_2,constellation_symbols); %decoded symbols

    decoded_bits_2 = symbol2bits_mapping(decoded_sym_2,constellation_symbols);%decoded bits sequence

    bits_in_error_2 = nnz(decoded_bits_2 ~= Tx_bits_2); % no of received bits in error 

    BER_2(i) = bits_in_error_2/no_of_bits; % BER 
end

%% Plot results
figure();
semilogy(0:1:10, BER_2, 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 10);
grid on; hold on;
semilogy(0:1:10, QAM_4_ML_BER, 'LineWidth', 2, 'Marker', '+', 'MarkerSize', 10)
legend("Neural detector - 2", "ML detector");
title("Setup 2")
xlabel("SNR (in dB)")
ylabel("BER")
set(findall(gcf,'-property','FontSize'),'FontSize',24)

%% saving the model parameters
%model_ouput_2 = net_2;
%save model_output_2