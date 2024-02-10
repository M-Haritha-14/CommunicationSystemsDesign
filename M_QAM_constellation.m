function symbols = M_QAM_constellation(M)
    %square M-QAM constellation
    m = -sqrt(M)/2+1:sqrt(M)/2;  %centering around origin
    x_in = 2*m-1; % computing in phase component
    Z = meshgrid(x_in,x_in); %since inphase and quadrature components are from same alphabet
    symbols = reshape(Z-1i*Z',1,[]); %M-QAM constellation symbols
end