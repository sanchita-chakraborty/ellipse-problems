function plot_optimal_ellipses(CHNK, C0, a_best, b_best, phi_best)
% CHNK: struct with CHNK.polyX, CHNK.polyY
% C0: N×2 centers
% a_best, b_best, phi: 1×N (or N×1) arrays

    assert(isfield(CHNK,'polyX') && isfield(CHNK,'polyY'), 'CHNK must have polyX/polyY');

    % prepare
    a_best = a_best(:).'; b_best = b_best(:).'; phi = phi_best(:).';
    N = size(C0,1);
    assert(numel(a_best)==N && numel(b_best)==N && numel(phi_best)==N, 'Size mismatCHNK');

    clf; hold on; axis equal;
    % plot exterior boundary
    plot([CHNK.polyX(:); CHNK.polyX(1)], [CHNK.polyY(:); CHNK.polyY(1)], 'k-', 'LineWidth', 1.25);

    % ellipse params
    th = linspace(0, 2*pi, 400);
    cth = cos(th); sth = sin(th);

    % draw eaCHNK ellipse
    for k = 1:N
        ct = cos(phi(k)); st = sin(phi(k));
        R  = [ct -st; st ct];
        XY = R * [a_best(k)*cth; b_best(k)*sth];   % 2×M
        xk = C0(k,1) + XY(1,:);
        yk = C0(k,2) + XY(2,:);
        plot(xk, yk, 'r-', 'LineWidth', 1.1);
    end

    % centers (optional)
    plot(C0(:,1), C0(:,2), 'r.', 'MarkerSize', 10);

    title('$\textbf{Optimal ellipses on exterior geometry}$', ...
          'Interpreter','latex');
    xlabel('x'); ylabel('y'); box on; hold off;
end
