% ML decoder decodes in favour of symbol with min euclidean distance
function decoded_symbol = ML_decoder(received_sym,constellation_symbols)

distance = abs(received_sym - (constellation_symbols).');

[~,idx] = min(distance);

decoded_symbol = constellation_symbols(idx);

end
