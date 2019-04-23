% demo.m
%
% Reduction of Poisson noise in coded exposure photography
%
% An ideal image 'x' is degraded by fluttered motion blur 'h' and mixed Poisson-Gaussian noise.
%
% Written by  : Chihiro Tsutake
% Affiliation : University of Fukui
% E-mail      : ctsutake@icloud.com
% Created     : April 2019
%

% Standard deviation of Gaussian noise
SIGMA = 3;
% Positive parameter for Poisson noise
ALPHA = 0.5;
% Positive parameter for OMEGA
TAU = 0.06;
% Positive parameter for l2-norm constraint
EPS = 8E+6;
% Convergence criterion
DLT = 1E-4;

% Read ideal image
x_idl = double(imread('img/a.png'));

% Shutter pattern
b_sht = double('1111111111111111111111111111110101010110101111111111') -'0';

% Blur impulse response
h_pad = zeros(size(x_idl));
h_pad(1, 1:52) = b_sht / 52;

% Observation
X_idl = fft2(x_idl);
H_pad = fft2(h_pad);
y_obs = ALPHA * poissrnd(real(ifft2(X_idl .* H_pad)) / ALPHA) + randn(size(x_idl)) * SIGMA;

% Set Omega
OMEGA = find(abs(H_pad) >= TAU);

% Partial inversion
X_hat = zeros(size(x_idl)); X_hat(OMEGA) = fft2(y_obs)(OMEGA) ./ H_pad(OMEGA);

% Reconstruction
x_hat = admm(OMEGA, X_hat, EPS, DLT);

% Write images
imwrite(uint8(y_obs), 'obs.png'); % observed
imwrite(uint8(x_hat), 'res.png'); % restored

% PSNR
printf("PSNR(x_idl, x_hat) = %4.2f\n", psnr(uint8(x_idl), uint8(x_hat)));


