clc;
clear all;
close all;

message_1 = zeros(11, 50000, 1); % initializing message array

message_2 = zeros(11, 50000, 1); % initializing message array

noisy_message = zeros(11, 50000, 1); % initializing noisy message array

M = 4; % constellation order

constellation_symbols = M_QAM_constellation(M); % fetching 4-QAM symbols

Es = ((norm(constellation_symbols))^2)/M; % average symbol energy

SNR_dB = 0:1:10; %SNR in dB scale
    
SNR = 10.^(SNR_dB/10); % SNR in linear scale

no_of_bits = 1e5; %samples

Tx_bits = randi([0 1],[1,no_of_bits]); % random sequence of 1's 0's

Tx_crumb_decimal = (bit2int(Tx_bits.',2))'; % converting each crumb(2 bits) to decimal equivalent

Tx_symbol = bits2symbol_mapping(Tx_crumb_decimal,constellation_symbols); % mapping the bits to symbols

SNR_cnt = numel(SNR_dB); % count of SNR's 

% AWGN noise vector
n = randn([1,length(Tx_symbol)]) + 1i*randn([1,length(Tx_symbol)]);

BER = zeros([1,SNR_cnt]);   %initializing BER array

for i = 1:1:SNR_cnt

    noise = sqrt(Es/(2*SNR(i))).*n; %varying noise power as 1/SNR

    message_1(SNR_dB(i)+1, :, :) = Tx_symbol; % true message for model 1

    message_2(SNR_dB(i)+1, :, :) = Tx_crumb_decimal; % true message for model 2

    Rx_symbol = Tx_symbol + noise; % Tx symbols over AWGN

    noisy_message(SNR_dB(i)+1, :, :) = Rx_symbol; % noisy message

    decoded_sym =  ML_decoder(Rx_symbol,constellation_symbols); %decoded symbols

    decoded_bits = symbol2bits_mapping(decoded_sym,constellation_symbols);%decoded bits sequence

    bits_in_error = nnz(decoded_bits ~= Tx_bits); % no of received bits in error 

    BER(i) = bits_in_error/no_of_bits; % BER
end

QAM_4_ML_BER = BER; % BER values for 4-QAM

% splitting dataset into training & validation with splitting ratio as 0.7

%% for model-1
% message for training
message_train_real = reshape(real(message_1(:, 1:35000, :)), [1, 11*35000]);
message_train_imag = reshape(imag(message_1(:, 1:35000, :)), [1, 11*35000]);
message_train_1 = [message_train_real;message_train_imag]; % each message is 2D vector

% noisy message for training
noisy_message_train_real = reshape(real(noisy_message(:, 1:35000, :)), [1, 11*35000]);
noisy_message_train_imag = reshape(imag(noisy_message(:, 1:35000, :)), [1, 11*35000]);
noisy_message_train = [noisy_message_train_real;noisy_message_train_imag]; % each noisy message is 2D vector

% message for validation
message_val_real = reshape(real(message_1(:, 35001:end, :)), [1, 11*15000]);
message_val_imag = reshape(imag(message_1(:, 35001:end, :)), [1, 11*15000]);
message_val_1 = [message_val_real;message_val_imag];

% noisy message for validation
noisy_message_val_real = reshape(real(noisy_message(:, 35001:end, :)), [1, 11*15000]);
noisy_message_val_imag = reshape(imag(noisy_message(:, 35001:end, :)), [1, 11*15000]);
noisy_message_val = [noisy_message_val_real;noisy_message_val_imag];

%% for model-2

message_train = reshape(message_2(:, 1:35000, :), [1, 11*35000]);
message_train_2 = categorical(message_train);

message_val = reshape((message_2(:, 35001:end, :)), [1, 11*15000]);
message_val_2 = categorical(message_val);

%%
%save('data_file.mat','QAM_4_ML_BER','message_train',"noisy_message_train","message_val","noisy_message_val","message_train_2","message_val_2");

