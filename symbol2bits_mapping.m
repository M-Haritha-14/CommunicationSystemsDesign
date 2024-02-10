function decoded_bits = symbol2bits_mapping(decoded_symbol,constellation_symbols)

symbols_MQAM_4_decimal = [0 1 2 3];  %4-QAM with gray code mapping

y = [];

for i = 1:1:length(decoded_symbol)

    sym_idx = find(constellation_symbols == decoded_symbol(:,i));
   
    sym_dec = symbols_MQAM_4_decimal(sym_idx);

    y = [y ; int2bit(sym_dec,2)];
   
end

decoded_bits = y.';

end


