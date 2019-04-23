% admm.m
%
% Solve quadratically constrained TV minimization problem
%
% OMEGA - Fourier-domain indices
% X_hat - partial DFT coefficients
% eps   - constraint
% dlt   - tolerance
%
% Written by  : Chihiro Tsutake
% Affiliation : University of Fukui
% E-mail      : ctsutake@icloud.com
% Created     : April 2019
%

function x_hat = admm(OMEGA, X_hat, eps, dlt)

    [H W] = size(X_hat);
    x_hat = zeros(H, W);

    % Variables
    prm_u = randn(H, W);
    aux_x = zeros(H, W);
    aux_y = zeros(H, W);
    aux_z = zeros(H, W);
    lag_a = zeros(H, W);
    lag_b = zeros(H, W);
    lag_c = zeros(H, W);

    % Kernel (Fourier domain)
    ker = zeros(H, W);
    ker([end, 1, 2], [end, 1, 2]) = [0, -1, 0; -1, 4, -1; 0, -1, 0];
    ker = real(fft2(ker));
    ker(OMEGA) = ker(OMEGA) + 1;

    % Complement of OMEGA
    cOMEGA = ones(H, W);
    cOMEGA(OMEGA) = 0;
    cOMEGA = find(cOMEGA);

    for i = 1:65536
        % Copy old
        old_u = prm_u;
        
        % Update primal variable
        tmp1 = aux_x - lag_a; tmp1 = -tmp1 + [tmp1(:, W), tmp1(:, 1:W - 1)];
        tmp2 = aux_y - lag_b; tmp2 = -tmp2 + [tmp2(H, :); tmp2(1:H - 1, :)];
        tmp3 = aux_z - lag_c + X_hat;
        prm_u = real(ifft2((fft2(tmp1 + tmp2) + tmp3) ./ ker));

        % Update auxiliary variables x and y
        tmp1 = -prm_u + [prm_u(:, 2:W), prm_u(:, 1)];
        tmp2 = -prm_u + [prm_u(2:H, :); prm_u(1, :)];
        tmp3 = tmp1 + lag_a;
        tmp4 = tmp2 + lag_b;
        th = sqrt(tmp3.^2 + tmp4.^2);
        aux_x = max(1 - 1 ./ th, 0) .* tmp3;
        aux_y = max(1 - 1 ./ th, 0) .* tmp4;

        % Update auxiliary variable z
        tmp3 = fft2(prm_u);
        tmp3(cOMEGA) = 0;
        tmp4 = tmp3 + lag_c - X_hat;
        l2 = norm(tmp4(:));
        aux_z = min(l2, eps) * (tmp4 / l2);

        % Update Lagrangian multiplier
        lag_a = lag_a + tmp1 - aux_x;
        lag_b = lag_b + tmp2 - aux_y;
        lag_c = lag_c + tmp3 - aux_z - X_hat;

        % Convergence criterion
        if norm(prm_u(:) - old_u(:)) / norm(old_u(:)) < dlt
            break;
        end

    end

    x_hat = prm_u;
