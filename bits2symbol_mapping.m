function Tx_symbol = bits2symbol_mapping(Tx_nibble_decimal,constellation_symbols)

symbols_MQAM_4_decimal = [0 1 2 3]; %4-QAM with gray code mapping

Tx_symbol = [];

for i = 1:1:length(Tx_nibble_decimal)

    sym_idx = find(symbols_MQAM_4_decimal == Tx_nibble_decimal(:,i));

    Tx_symbol = [Tx_symbol constellation_symbols(sym_idx)];
end

end