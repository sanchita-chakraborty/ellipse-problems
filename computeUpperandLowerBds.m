%% computeUpperandLowerBds
% ===== proportions config =====
% Global min/max of normalized semi-axes across ALL cells
S = load('fly_muscle_data.mat');
need = {'edgeX','edgeY','NucMajor','NucMinor','startPoints','endPoints'};
for k=1:numel(need), assert(isfield(S,need{k}), 'Missing %s', need{k}); end

alpha_a_all = []; alpha_b_all = [];
nCells = size(S.edgeX,1);

for c = 1:nCells
    X = S.edgeX(c,:).'; Y = S.edgeY(c,:).';
    g = isfinite(X) & isfinite(Y); X=X(g); Y=Y(g);
    if numel(X) < 3, continue; end
    if X(1)~=X(end) || Y(1)~=Y(end), X=[X;X(1)]; Y=[Y;Y(1)]; end
    A = polyarea(X,Y); if ~(isfinite(A) && A>0), continue; end
    Lc = sqrt(A);                         % cell length scale

    i0 = S.startPoints(c); i1 = S.endPoints(c);
    a = S.NucMajor(i0:i1)/2;              % semi-major
    b = S.NucMinor(i0:i1)/2;              % semi-minor
    ok = isfinite(a)&isfinite(b)&a>0&b>0;
    if ~any(ok), continue; end

    alpha_a_all = [alpha_a_all; a(ok)/Lc]; %#ok<AGROW>
    alpha_b_all = [alpha_b_all; b(ok)/Lc]; %#ok<AGROW>
end

alpha_a_all = alpha_a_all(isfinite(alpha_a_all));
alpha_b_all = alpha_b_all(isfinite(alpha_b_all));

alpha_a_min = min(alpha_a_all);
alpha_a_max = max(alpha_a_all);
alpha_b_min = min(alpha_b_all);
alpha_b_max = max(alpha_b_all);

fprintf('Proportion bounds (ALL cells):\n');
fprintf('  a/L: [%.6g, %.6g]\n', alpha_a_min, alpha_a_max);
fprintf('  b/L: [%.6g, %.6g]\n', alpha_b_min, alpha_b_max);

nCells = size(S.edgeX,1);
all_pairs_norm = [];         % collect all pairwise distances / Lc
nnorm_nearest  = [];         % collect nearest-neighbor distances / Lc

for c = 1:nCells
    % --- cell boundary area & scale ---
    X = S.edgeX(c,:).'; Y = S.edgeY(c,:).';
    g = isfinite(X) & isfinite(Y); X=X(g); Y=Y(g);
    if numel(X) < 3, continue; end
    if X(1)~=X(end) || Y(1)~=Y(end), X=[X;X(1)]; Y=[Y;Y(1)]; end
    A  = polyarea(X,Y); if ~(isfinite(A) && A>0), continue; end
    Lc = sqrt(A);

    % --- nuclei for this cell ---
    i0 = S.startPoints(c); i1 = S.endPoints(c);
    nx = S.NucX(i0:i1); ny = S.NucY(i0:i1);
    ok = isfinite(nx) & isfinite(ny);
    nx = nx(ok); ny = ny(ok);
    N  = numel(nx);
    if N < 2, continue; end

    % --- pairwise distances within this cell ---
    % use pdist if available, else manual
    if exist('pdist','file') == 2
        d = pdist([nx ny]);                 % vector of all pairwise distances
        d = d(:) / Lc;                      % normalize by Lc
    else
        % manual upper triangle
        d = [];
        for i=1:N-1
            dx = nx(i) - nx(i+1:N);
            dy = ny(i) - ny(i+1:N);
            d  = [d; hypot(dx,dy)]; %#ok<AGROW>
        end
        d = d / Lc;
    end
    all_pairs_norm = [all_pairs_norm; d];    %#ok<AGROW>

    % --- nearest-neighbor distance per nucleus (exclude self) ---
    % efficient: compute row-wise mins
    % small N: use full matrix
    D = hypot(nx - nx.', ny - ny.');
    D(1:N+1:end) = inf;                      % ignore self
    nn = min(D, [], 2) / Lc;                 % nearest neighbor per point
    nnorm_nearest = [nnorm_nearest; nn];     %#ok<AGROW>
end

% Clean and summarize
all_pairs_norm = all_pairs_norm(isfinite(all_pairs_norm) & all_pairs_norm>=0);
nnorm_nearest  = nnorm_nearest(isfinite(nnorm_nearest) & nnorm_nearest>=0);

if isempty(all_pairs_norm) || isempty(nnorm_nearest)
    error('No valid distances found across dataset.');
end

% === FOUR NUMBERS (all-pairs) ===
d_pair_min = min(all_pairs_norm);
d_pair_max = max(all_pairs_norm);

% === FOUR NUMBERS (nearest-neighbor) ===
d_nn_min = min(nnorm_nearest);
d_nn_max = max(nnorm_nearest);

fprintf('Normalized distance proportions across ALL cells:\n');
fprintf('  ALL-PAIRS  d/L: [%.6g, %.6g]\n', d_pair_min, d_pair_max);
fprintf('  NEAREST-NB d/L: [%.6g, %.6g]\n', d_nn_min,   d_nn_max);

% sampling resolution on ellipse boundary
Mth = 180; th = linspace(0,2*pi,Mth+1); th(end) = [];

% accumulators (normalized)
global_point_min = +inf;   % min over ALL sampled points
global_point_max = -inf;   % max over ALL sampled points

min_clear_norm_all = [];   % per-ellipse min(clear)/Lc
max_clear_norm_all = [];   % per-ellipse max(clear)/Lc

nCells = size(S.edgeX,1);

for c = 1:nCells
    % --- cell polygon ---
    X = S.edgeX(c,:).'; Y = S.edgeY(c,:).';
    g = isfinite(X) & isfinite(Y); X = X(g); Y = Y(g);
    if numel(X) < 3, continue; end
    if X(1)~=X(end) || Y(1)~=Y(end), X = [X; X(1)]; Y = [Y; Y(1)]; end
    A = polyarea(X,Y); if ~(isfinite(A) && A>0), continue; end
    Lc = sqrt(A);

    % prebuild polygon segments
    segs = [X(1:end-1) Y(1:end-1) X(2:end) Y(2:end)];

    % --- nuclei in this cell ---
    i0 = S.startPoints(c); i1 = S.endPoints(c);
    cx = S.NucX(i0:i1); cy = S.NucY(i0:i1);
    a  = S.NucMajor(i0:i1)/2; b = S.NucMinor(i0:i1)/2;
    ph = S.NucAngle(i0:i1);
    ok = isfinite(cx)&isfinite(cy)&isfinite(a)&isfinite(b)&isfinite(ph)&a>0&b>0;
    cx = cx(ok); cy = cy(ok); a = a(ok); b = b(ok); ph = ph(ok);
    N  = numel(cx); if N==0, continue; end

    % --- for each ellipse, sample boundary and compute distance to polygon ---
    for i = 1:N
        ct = cos(ph(i)); st = sin(ph(i));
        XY = [a(i)*cos(th); b(i)*sin(th)];
        R  = [ct -st; st ct];
        P  = R*XY + [cx(i); cy(i)];   % 2×Mth ellipse boundary points

        % distances from each boundary point to polygon (min over segments)
        d = inf(1,Mth);
        for m = 1:Mth
            p = P(:,m).';
            d(m) = min_point_to_segments(p, segs);
        end
        d_norm = d / Lc;

        % update global pointwise bounds
        global_point_min = min(global_point_min, min(d_norm));
        global_point_max = max(global_point_max, max(d_norm));

        % record per-ellipse min/max (normalized)
        min_clear_norm_all(end+1,1) = min(d_norm); %#ok<AGROW>
        max_clear_norm_all(end+1,1) = max(d_norm); %#ok<AGROW>
    end
end

% ---- final bounds ----
d_point_norm_min = global_point_min;
d_point_norm_max = global_point_max;

% per-ellipse summaries aggregated across all ellipses
min_over_ellipses_of_min_clear_norm = min(min_clear_norm_all);
max_over_ellipses_of_min_clear_norm = max(min_clear_norm_all);
min_over_ellipses_of_max_clear_norm = min(max_clear_norm_all);
max_over_ellipses_of_max_clear_norm = max(max_clear_norm_all);

fprintf('Pointwise clearance proportion d/L across ALL points: [%.6g, %.6g]\n', ...
        d_point_norm_min, d_point_norm_max);
fprintf(['Per-ellipse min(clear)/L across ellipses:   [min=%.6g, max=%.6g]\n' ...
         'Per-ellipse max(clear)/L across ellipses:   [min=%.6g, max=%.6g]\n'], ...
        min_over_ellipses_of_min_clear_norm, max_over_ellipses_of_min_clear_norm, ...
        min_over_ellipses_of_max_clear_norm, max_over_ellipses_of_max_clear_norm);

% --- helper: distance from point p to a polyline given as segment list ---
function d = min_point_to_segments(p, segs)
    % segs: M×4, each row [x1 y1 x2 y2]
    ax = segs(:,1); ay = segs(:,2);
    bx = segs(:,3); by = segs(:,4);
    apx = p(1) - ax; apy = p(2) - ay;
    abx = bx - ax;   aby = by - ay;
    ab2 = abx.^2 + aby.^2;
    t   = (apx.*abx + apy.*aby) ./ max(ab2, eps);
    t   = max(0, min(1, t));
    projx = ax + t.*abx;  projy = ay + t.*aby;
    d = min( hypot(p(1)-projx, p(2)-projy) );
end