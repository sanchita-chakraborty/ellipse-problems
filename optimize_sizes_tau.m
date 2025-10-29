function [a_best, b_best, phi_best, tau_best, hist] = optimize_sizes_tau(CH, C_best, varargin)
% Optimize ellipse sizes (a,b) and orientations phi with centers fixed at C_best.
% Physics objective via SolveAsy (uses CHNK/CH cache for Green's).
%
% Inputs:
%   CH       : struct with .ell, .polyX, .polyY, .area, .chnkr, ...
%   C_best   : N×2 centers (x,y)
%
% Options (Name/Value):
%   'UseN'        : use first N ellipses (default all)
%   'AB0'         : initial [a;b] (2×N or N×2); default CH.ell.{a,b}
%   'SwarmSize'   : PSO swarm size (default 30)
%   'MaxIters'    : PSO iterations (default 30)
%   'PenaltyW'    : global penalty weight (default 1e3)
%   'BndMargin'   : clearance (in pixels) from polygon boundary (default 0.04)
%   'SepMargin'   : extra separation between ellipses (kept for compatibility; not used)
%   'PropALB'     : lower bound proportion for a (a/L) e.g. 0.02
%   'PropAUB'     : upper bound proportion for a (a/L) e.g. 0.11
%   'PropBLB'     : lower bound proportion for b (b/L) e.g. 0.02
%   'PropBUB'     : upper bound proportion for b (b/L) e.g. 0.07
%   'DnnLB'       : lower band for d_nn/L (default 0.05)
%   'DnnUB'       : upper band for d_nn/L (default 1.3)
%   'WSizeProp'   : weight for boundary clearance penalty (default 1e3)
%   'WNNProp'     : weight for NN band penalty (default 1e3)
%   'PropPNorm'   : hinge power (1 or 2) (default 2)
%   'Plot'        : live plot (default false)
%   'TauFcn'      : @(o,eps_scalar)-> scalar tau (default uses o.tau or chi0+eps^2 chi2)
%   'ExcelFile'   : progress xlsx (default 'AB_progress.xlsx')
%   'ExcelSheet'  : sheet name (default 'progress')
%   'LogEvery'    : log every k iterations (default 5)
%
% Outputs:
%   a_best, b_best, phi_best : 1×N
%   tau_best                 : best penalized objective
%   hist.best (per iter), hist.feasible (per iter)

% -------- options --------
p = inputParser;
p.addParameter('UseN', []);
p.addParameter('AB0', []);
p.addParameter('SwarmSize', 30);
p.addParameter('MaxIters', 30);
p.addParameter('PenaltyW', 1e3);
p.addParameter('BndMargin', 0.04);
p.addParameter('SepLB', 0.05);      % lower bound on normalized separation
p.addParameter('SepUB', 1.30);       % upper bound on normalized separation
p.addParameter('SepMode','nn');      % 'nn' (nearest-neighbor) or 'all' (all-pairs)
p.addParameter('WSep', 1e3);         % weight for separation band penalty
p.addParameter('PropALB', 0.02);
p.addParameter('PropAUB', 0.11);
p.addParameter('PropBLB', 0.02);
p.addParameter('PropBUB', 0.07);
p.addParameter('DnnLB',   0.05);
p.addParameter('DnnUB',   1.3);
p.addParameter('WSizeProp', 1e3);
p.addParameter('WNNProp',   1e3);
p.addParameter('PropPNorm', 2);
p.addParameter('Plot', false);
p.addParameter('TauFcn', []);
p.addParameter('ExcelFile', 'AB_progress.xlsx');
p.addParameter('ExcelSheet','progress');
p.addParameter('LogEvery', 5);
p.parse(varargin{:});
opt = p.Results;
Wpen = opt.PenaltyW;

% -------- setup --------
assert(isfield(CH,'ell') && isfield(CH,'polyX') && isfield(CH,'polyY') && isfield(CH,'area'), ...
    'CH must have .ell, .polyX, .polyY, .area');

% Cell length scale for proportions (use sqrt(area) )
L0 = sqrt(polyarea(CH.polyX(:), CH.polyY(:)));
assert(isfinite(L0) && L0>0, 'Bad cell scale L0');

E    = CH.ell;
Nall = numel(E.cx);
N    = Nall;
if ~isempty(opt.UseN), N = min(Nall, opt.UseN); end

C0   = C_best(1:N,:);        % fixed centers
a0   = E.a(1:N).';
b0   = E.b(1:N).';
phi0 = E.phi(1:N).';

% If AB0 provided, use it
if ~isempty(opt.AB0)
    AB0 = reshape(opt.AB0,[],N);
    a0 = AB0(1,:).';
    b0 = AB0(2,:).';
end

% ---------- HARD PSO BOUNDS (proportion-based for a,b; full range for phi) ----------
a_lb0 = (opt.PropALB * L0) * ones(N,1);
a_ub0 = (opt.PropAUB * L0) * ones(N,1);
b_lb0 = (opt.PropBLB * L0) * ones(N,1);
b_ub0 = (opt.PropBUB * L0) * ones(N,1);

% tighten by local clearance (conservative)
clear_i = local_clearance_to_polygon(C0, CH.polyX, CH.polyY); % N×1
a_ub = min(a_ub0, max(1e-9, clear_i - opt.BndMargin));
b_ub = min(b_ub0, max(1e-9, clear_i - opt.BndMargin));

phi_lb = (-pi)*ones(N,1);
phi_ub = ( pi)*ones(N,1);

% PSO bounds (row vectors)
LB = [a_lb0.' , b_lb0.' , phi_lb.'];
UB = [a_ub .' , b_ub .' , phi_ub .'];

% initial guess and clamp
x0 = [a0.' , b0.' , phi0.'];
x0 = max(min(x0, UB), LB);
x0 = x0(:).';

% -------- history --------
hist.best     = +inf(opt.MaxIters,1);
hist.feasible = false(opt.MaxIters,1);
iter_counter  = 0;
tau_best      = +inf;
a_best        = a0.'; 
b_best        = b0.'; 
phi_best      = phi0.';

TauFcn = opt.TauFcn;

% ============================ helpers ============================

% evaluate tau for given a,b,phi (centers fixed)
function tau = eval_tau(aa, bb, pphi)
    % wrap angles for numerical stability
    pphi = atan2(sin(pphi), cos(pphi));

    % eps from geometry size
    Lx = range(CH.chnkr.r(1,:));  Ly = range(CH.chnkr.r(2,:));
    Lref = max([Lx, Ly, sqrt(CH.area)]);
    eps_scalar = 1./max(Lref, eps);

    o = struct();
    o.Nc     = N;
    o.x      = C0(:,1).';
    o.y      = C0(:,2).';
    o.a      = aa(:).';
    o.b      = bb(:).';
    o.phi    = pphi(:).';
    o.eps    = eps_scalar * ones(1,N);
    o.area   = CH.area;
    o.x_eval = o.x(:); 
    o.y_eval = o.y(:);

    DO_PARFOR = false;
    o = SolveAsy(o, DO_PARFOR);

    if ~isempty(TauFcn)
        tau = TauFcn(o, eps_scalar);
    else
        if isfield(o,'tau') && ~isempty(o.tau)
            tau = o.tau;
        else
            tau = o.chi0 + (eps_scalar^2)*o.chi2;
        end
    end
end

% write progress to Excel every k iters
function write_progress(k, tau_k, aa, bb, pp)
    if opt.LogEvery <= 0 || mod(k,opt.LogEvery) ~= 0, return; end
    fn = opt.ExcelFile; sh = opt.ExcelSheet;

    if k == opt.LogEvery
        hdr = ["iter","tau", strcat("a",string(1:N)), strcat("b",string(1:N)), strcat("phi",string(1:N))];
        try, writematrix(hdr, fn, 'Sheet', sh, 'Range','A1'); catch, end
    end

    row = k/opt.LogEvery + 1;
    rowdata = [k, tau_k, aa(:).', bb(:).', pp(:).'];
    tries=0; ok=false;
    while ~ok && tries<3
        try
            writematrix(rowdata, fn, 'Sheet', sh, 'Range', ['A' num2str(row)]);
            ok=true;
        catch
            tries=tries+1; pause(0.25);
        end
    end
end

% ============================ objective =========================
function f = wrapper(psx)
    % ===== unpack PSO vector =====
    v   = reshape(psx,[],1);
    aa  = v(1:N);
    bb  = v(N+1:2*N);
    pph = v(2*N+1:3*N);

    % ===== robustness clamps =====
    aa = max(aa, 1e-9);
    bb = max(bb, 1e-9);

    % ===== 1) physics =====
    tau_raw = eval_tau(aa, bb, pph);

    % ===== 2) penalties =====
    pwr = max(1, opt.PropPNorm);

    % ---- boundary clearance: (max(a,b)+margin) <= center-clearance ----
    clr_center = local_clearance_to_polygon(C0, CH.polyX, CH.polyY);  % N×1
    need_gap   = max(aa, bb) + opt.BndMargin;
    penBnd     = sum( max(0, (need_gap - clr_center)/L0 ).^pwr );

    % ---- separation band options (safe defaults without getfield) ----
    if isfield(opt,'SepLB') && ~isempty(opt.SepLB)
        SepLB = opt.SepLB;
    elseif isfield(opt,'DnnLB') && ~isempty(opt.DnnLB)
        SepLB = opt.DnnLB;
    else
        SepLB = 0.05;
    end

    if isfield(opt,'SepUB') && ~isempty(opt.SepUB)
        SepUB = opt.SepUB;
    elseif isfield(opt,'DnnUB') && ~isempty(opt.DnnUB)
        SepUB = opt.DnnUB;
    else
        SepUB = 1.3;
    end

    if isfield(opt,'SepMode') && ~isempty(opt.SepMode)
        SepMode = opt.SepMode;           % 'nn' or 'all'
    else
        SepMode = 'nn';
    end

    if isfield(opt,'WSep') && ~isempty(opt.WSep)
        WSep = opt.WSep;
    elseif isfield(opt,'WNNProp') && ~isempty(opt.WNNProp)
        WSep = opt.WNNProp;
    else
        WSep = 1e3;
    end

    if isfield(opt,'SizeAwareSep') && ~isempty(opt.SizeAwareSep)
        SizeAwareSep = logical(opt.SizeAwareSep);
    else
        SizeAwareSep = false;
    end

    % ---- separation penalty on normalized distance ----
    DX   = C0(:,1) - C0(:,1).';
    DY   = C0(:,2) - C0(:,2).';
    Dmat = hypot(DX,DY);
    Dmat(1:N+1:end) = inf;  % ignore self

    if SizeAwareSep
        % boundary-to-boundary clearance (approx) using equivalent radii
        ri   = 0.5*(aa+bb);          % N×1
        rj   = 0.5*(aa'+bb');        % 1×N -> broadcast
        Deff = Dmat - (ri + rj);     % N×N
        Deff(1:N+1:end) = inf;

        if strcmpi(SepMode,'nn')
            d_use = min(Deff,[],2) / L0;      % N×1 nearest clearance
        else
            d_use = Deff(triu(true(N),1)) / L0;  % all unique pairs
        end
    else
        % center-to-center spacing only
        if strcmpi(SepMode,'nn')
            d_use = min(Dmat,[],2) / L0;          % N×1
        else
            d_use = Dmat(triu(true(N),1)) / L0;   % all unique pairs
        end
    end

    penSep = sum( max(0, SepLB - d_use).^pwr + max(0, d_use - SepUB).^pwr );

    % ---- combine penalties ----
    pen = opt.WSizeProp*penBnd + WSep*penSep;

    % ===== 3) penalized objective =====
    f = tau_raw + Wpen*pen;

    % ===== 4) bookkeeping =====
    iter_counter = iter_counter + 1;
    if f < tau_best
        tau_best = f;
        a_best   = aa.'; 
        b_best   = bb.';
        phi_best = atan2(sin(pph), cos(pph)).';
    end
    k = min(iter_counter, numel(hist.best));
    hist.best(k)     = tau_best;
    hist.feasible(k) = (pen==0);

    write_progress(k, tau_best, a_best, b_best, phi_best);

    % ===== optional plot =====
    if opt.Plot && mod(k,10)==0
        clf; hold on; axis equal;
        plot([CH.polyX CH.polyX(1)], [CH.polyY CH.polyY(1)], 'k-', 'LineWidth', 1.1);
        th = linspace(0,2*pi,200);
        for t=1:N
            ct = cos(phi_best(t)); st = sin(phi_best(t));
            XY = [a_best(t)*cos(th); b_best(t)*sin(th)];
            xy = [ct -st; st ct]*XY;
            plot(C0(t,1)+xy(1,:), C0(t,2)+xy(2,:), 'r-');
        end
        title(sprintf('iter %d | f_{best}=%.6g', k, tau_best));
        drawnow;
    end
end

% ============================ run PSO ===========================
nvar = 3*N;
if exist('particleswarm','file')==2
    opts = optimoptions('particleswarm','SwarmSize',opt.SwarmSize, ...
        'MaxIterations',opt.MaxIters,'Display','iter','UseParallel',false);
    particleswarm(@wrapper, nvar, LB, UB, opts);
else
    % Simple PSO-lite fallback
    S=opt.SwarmSize; W=0.72; c1=1.5; c2=1.5;
    rng('shuffle');
    X = repmat(x0,S,1) + 0.2*(randn(S,nvar).*repmat(UB-LB,S,1));
    X = max(min(X,UB),LB);
    V = zeros(S,nvar);
    pX = X; pF = inf(S,1);
    for it=1:opt.MaxIters
        for s=1:S
            f = wrapper(X(s,:));
            if f < pF(s), pF(s)=f; pX(s,:)=X(s,:); end
        end
        [~,gi]=min(pF); gX=pX(gi,:);
        r1=rand(S,nvar); r2=rand(S,nvar);
        V = W*V + c1*r1.*(pX-X) + c2*r2.*(repmat(gX,S,1)-X);
        X = max(min(X + V, UB), LB);
    end
end
end

% ============================ utilities =========================
function dmin = local_clearance_to_polygon(C, polyX, polyY)
    % min distance from each center to polygon boundary (piecewise segments)
    if polyX(1)~=polyX(end) || polyY(1)~=polyY(end)
        polyX=[polyX(:); polyX(1)]; polyY=[polyY(:); polyY(1)];
    else
        polyX=polyX(:); polyY=polyY(:);
    end
    segs = [polyX(1:end-1) polyY(1:end-1) polyX(2:end) polyY(2:end)];
    N = size(C,1); dmin = inf(N,1);
    for i=1:N
        p = C(i,:);
        d = inf;
        for s=1:size(segs,1)
            d = min(d, point_segment_dist(p, segs(s,1:2), segs(s,3:4)));
        end
        dmin(i) = d;
    end
end

function d = point_segment_dist(p, a, b)
    ap = p - a; ab = b - a;
    t = max(0, min(1, dot(ap,ab)/max(dot(ab,ab),eps)));
    proj = a + t*ab;
    d = hypot(p(1)-proj(1), p(2)-proj(2));
end


% -------- Excel logger (size progress) --------
function write_progress(k, tau_k, aa, bb)
    if opt.LogEvery <= 0, return; end
    if mod(k, opt.LogEvery) ~= 0, return; end
    fn = opt.ExcelFile; sh = opt.ExcelSheet;

    if k == opt.LogEvery
        hdr = ["iter","tau"];
        xyhdr = strings(1, 2*N);
        for t=1:N, xyhdr(2*t-1:2*t) = ["a"+t, "b"+t]; end
        try, writematrix([hdr, xyhdr], fn, 'Sheet', sh, 'Range','A1'); catch, end
    end

    row = k/opt.LogEvery + 1;
    rowdata = [k, tau_k, aa(:).', bb(:).'];
    tries=0; ok=false;
    while ~ok && tries<3
        try
            writematrix(rowdata, fn, 'Sheet', sh, 'Range', ['A' num2str(row)]);
            ok=true;
        catch
            tries=tries+1; pause(0.25);
        end
    end
end

function o = SolveAsy(o, DO_PARFOR) %#ok<INUSD>
% SOLVEASY  Asymptotic coefficients using dataset-backed Neumann Green's fn.
% - Builds G_mat in one batched call (nuclei-as-targets, nuclei-as-sources)
% - Solves leading order (S0, chi0) and correction (S2, chi2)
% - Precomputes vectors/mats used in u-field evaluation

N = numel(o.x);
G_mat = zeros(N);  %#ok<NASGU>   % overwritten below; kept for clarity

% ------------------ geometry-dependent small parameter -------------------
o.eps = o.eps(:).';  o.a = o.a(:).';  o.b = o.b(:).';
o.nu  = -1 ./ log( 0.5 .* o.eps .* (o.a + o.b) );

% ------------------ ellipse geometric tensors ---------------------------
o.Q_mats = zeros(2,2,N);
o.M_mats = zeros(2,2,N);
for k = 1:N
    o.Q_mats(:,:,k) = -0.25*(o.a(k)^2 - o.b(k)^2) * ...
                      [ cos(2*o.phi(k))  sin(2*o.phi(k));
                        sin(2*o.phi(k)) -cos(2*o.phi(k)) ];
    o.M_mats(:,:,k) = -0.25*(o.a(k)+o.b(k))^2 * eye(2) + o.Q_mats(:,:,k);
end

% ================== 1) Batch self-regulars + full Green matrix ==========
% All nuclei as targets against all nuclei as sources, with derivatives
Esrc = NeumGR_dataset(o.x(:), o.y(:), o.x(:), o.y(:), true);

% Pull self-regular parts (diagonal of R) and its grads/Hessian
idx = (1:N).';
lin = sub2ind([N,N], idx, idx);
R0      = Esrc.R(lin).';                        % 1×N
GR0     = [Esrc.Rx(lin).'; Esrc.Ry(lin).'];     % 2×N
HR0     = zeros(2,2,N);
HR0_xx  = Esrc.Rxx(lin).';  HR0_xy = Esrc.Rxy(lin).';
HR0_yx  = Esrc.Ryx(lin).';  HR0_yy = Esrc.Ryy(lin).';
for k=1:N, HR0(:,:,k) = [HR0_xx(k) HR0_xy(k); HR0_yx(k) HR0_yy(k)]; end

% Full Green matrix: off-diag = G, diag = R
G_mat = Esrc.G;
for k=1:N, G_mat(k,k) = R0(k); end
G_mat = (G_mat + G_mat.')/2;                    % gentle symmetrization

% ================== 2) Leading order system =============================
M = [ (eye(N) + 2*pi*G_mat*diag(o.nu)), -ones(N,1); ...
       o.nu,                             0          ];
rhs0 = [zeros(N,1); o.area/(2*pi)];
A = M\rhs0;
o.S0   = A(1:N);
o.chi0 = A(end);

% ================== 3) Correction-term pieces ===========================
o.b_vec  = zeros(2,N);
o.H_mats = zeros(2,2,N);

% Start from self-regular contributions...
for k = 1:N
    o.b_vec(:,k)    = -2*pi*o.S0(k)*o.nu(k)*GR0(:,k);
    o.H_mats(:,:,k) = -2*pi*o.S0(k)*o.nu(k)*HR0(:,:,k);
end

% ...then add contributions from other sources using batch derivatives of G
% For target row k, pull derivatives against all sources m
scaleSm = -2*pi * (o.S0(:)'.*o.nu(:)');  % 1×N, multiplies each column m

for k = 1:N
    gGx = Esrc.Gx(k,:);  gGy = Esrc.Gy(k,:);
    Hxx = Esrc.Gxx(k,:); Hxy = Esrc.Gxy(k,:);
    Hyx = Esrc.Gyx(k,:); Hyy = Esrc.Gyy(k,:);
    % zero self (m = k)
    gGx(k)=0; gGy(k)=0; Hxx(k)=0; Hxy(k)=0; Hyx(k)=0; Hyy(k)=0;

    % b_vec additions
    o.b_vec(1,k) = o.b_vec(1,k) + sum(scaleSm .* gGx);
    o.b_vec(2,k) = o.b_vec(2,k) + sum(scaleSm .* gGy);

    % H_mats additions (sum over m != k)
    o.H_mats(1,1,k) = o.H_mats(1,1,k) + sum(scaleSm .* Hxx);
    o.H_mats(1,2,k) = o.H_mats(1,2,k) + sum(scaleSm .* Hxy);
    o.H_mats(2,1,k) = o.H_mats(2,1,k) + sum(scaleSm .* Hyx);
    o.H_mats(2,2,k) = o.H_mats(2,2,k) + sum(scaleSm .* Hyy);
end

% ================== 4) RHS for correction (batched) =====================
rhs = zeros(N,1);
for k = 1:N
    GradXiR  = GR0(:,k).';
    HessXiR  = HR0(:,:,k);

    rhs_k = pi*o.S0(k)*o.nu(k)*trace(o.Q_mats(:,:,k)*HessXiR) ...
          + 2*pi*(o.b_vec(:,k)'*(o.M_mats(:,:,k)*GradXiR')) ...
          - (o.a(k)^2 + o.b(k)^2)/8 ...
          - 0.5*trace(o.Q_mats(:,:,k)*o.H_mats(:,:,k));

    % Add ∑_{m≠k} [ π S0(m)ν(m) tr(Q_m Hess_x G(x_k, x_m))
    %             + 2π b(m)^T M(m) ∇_x G(x_k, x_m) ]
    gGx = Esrc.Gx(k,:);  gGy = Esrc.Gy(k,:);
    Hxx = Esrc.Gxx(k,:); Hxy = Esrc.Gxy(k,:);
    Hyx = Esrc.Gyx(k,:); Hyy = Esrc.Gyy(k,:);
    gGx(k)=0; gGy(k)=0; Hxx(k)=0; Hxy(k)=0; Hyx(k)=0; Hyy(k)=0;

    % First term: tr(Q_m * HessG) summed over m
    qH = zeros(1,N);
    for m=1:N
        if m==k, continue; end
        Hm = [Hxx(m) Hxy(m); Hyx(m) Hyy(m)];
        qH(m) = trace( o.Q_mats(:,:,m) * Hm );
    end
    rhs_k = rhs_k + pi * sum( (o.S0(:)'.*o.nu(:)').* qH );

    % Second term: 2π b(m)^T M(m) ∇G, summed over m
    add2 = 0;
    for m=1:N
        if m==k, continue; end
        gradG = [gGx(m); gGy(m)];
        add2 = add2 + ( o.b_vec(:,m)' * ( o.M_mats(:,:,m) * gradG ) );
    end
    rhs_k = rhs_k + 2*pi*add2;

    rhs(k) = rhs_k;
end

A = M\[rhs; 0];
o.S2   = A(1:N);
o.chi2 = A(end);

% ================== 5) Field on eval points (coefficients) ==============
N_pts = numel(o.x_eval);
if N_pts>0
    Eev = NeumGR_dataset(o.x_eval(:), o.y_eval(:), o.x(:), o.y(:), true);

    % leading field
    o.u_asy0 = o.chi0*ones(N_pts,1) - 2*pi * ( Eev.G * (o.S0(:).*o.nu(:)) );

    % fix coincidences to use R instead of G
    for i=1:N
        mask = (abs(o.x_eval(:)-o.x(i))<1e-15) & (abs(o.y_eval(:)-o.y(i))<1e-15);
        if any(mask)
            o.u_asy0(mask) = o.u_asy0(mask) ...
                + 2*pi*o.S0(i)*o.nu(i) * ( Eev.G(mask,i) - Eev.R(mask,i) );
        end
    end

    % correction field
    o.u_asy2 = o.chi2*ones(N_pts,1);

    % term1:  π ∑_i S0(i)ν(i) tr(Q_i * HessG_i)
    trQH = zeros(N_pts,N);
    for i=1:N
        trQH(:,i) = o.Q_mats(1,1,i)*Eev.Gxx(:,i) + ...
                    (o.Q_mats(1,2,i)+o.Q_mats(2,1,i))*Eev.Gxy(:,i) + ...
                     o.Q_mats(2,2,i)*Eev.Gyy(:,i);
    end
    term1 = pi * ( trQH * (o.S0(:).*o.nu(:)) );

    % term2:  2π ∑_i b(i)^T M(i) ∇G_i
    MGx = zeros(N_pts,N);  MGy = zeros(N_pts,N);
    for i=1:N
        MGx(:,i) = o.M_mats(1,1,i)*Eev.Gx(:,i) + o.M_mats(1,2,i)*Eev.Gy(:,i);
        MGy(:,i) = o.M_mats(2,1,i)*Eev.Gx(:,i) + o.M_mats(2,2,i)*Eev.Gy(:,i);
    end
    term2 = 2*pi * ( MGx * o.b_vec(1,:).' + MGy * o.b_vec(2,:).' );

    % term3: -2π ∑_i S2(i)ν(i) G_i
    term3 = -2*pi * ( Eev.G * (o.S2(:).*o.nu(:)) );

    o.u_asy2 = o.u_asy2 + term1 + term2 + term3;
end
end

function E = NeumGR_dataset(Xeval, Yeval, Xsrc, Ysrc, want_derivs)
if nargin<5, want_derivs = true; end
CHNK = CHNK_get(); chnkr = CHNK.chnkr; area = CHNK.area; wts = chnkr.wts(:);
Sk = kernel('laplace','s'); Kgrad = kernel.lap2d('sgrad');
KH = kernel(); KH.eval = @hesslap_s_eval; KH.opdims=[4 1]; KH.sing='smooth';

Ne = numel(Xeval); Ns = numel(Xsrc);
xe = Xeval(:); ye = Yeval(:); xs = Xsrc(:); ys = Ysrc(:);
targ = struct('r',[xe.';ye.']);

E.R   = zeros(Ne,Ns);  E.G   = zeros(Ne,Ns);
if want_derivs
  E.Rx  = zeros(Ne,Ns); E.Ry  = zeros(Ne,Ns);
  E.Rxx = zeros(Ne,Ns); E.Rxy = zeros(Ne,Ns); E.Ryx = zeros(Ne,Ns); E.Ryy = zeros(Ne,Ns);
  E.Gx  = zeros(Ne,Ns); E.Gy  = zeros(Ne,Ns);
  E.Gxx = zeros(Ne,Ns); E.Gxy = zeros(Ne,Ns); E.Gyx = zeros(Ne,Ns); E.Gyy = zeros(Ne,Ns);
end

for i = 1:Ns
  pack  = build_sigma_for_source([xs(i); ys(i)]);
  sigma = pack.sigma;

  Ssig = chunkerkerneval(chnkr, Sk, sigma, targ);        % Ne×1
  poly = (xe.^2 + ye.^2)/(4*area);
  wbar = sum(sigma .* wts);                              % scalar
  Ri   = Ssig + poly + wbar;                             % regular part
  E.R(:,i) = Ri;

  Ssrc = chnk.lap2d.kern(struct('r',[xs(i);ys(i)]), targ, 's'); % singular kernel
  E.G(:,i) = Ri + Ssrc - wbar;                           % full Green (off-diag)

  if want_derivs
    J = chunkerkerneval(chnkr, Kgrad, sigma, targ);      % (2×Ne)
    gradSsig = reshape(J,2,[]);
    Rx = gradSsig(1,:).' + xe/(2*area);
    Ry = gradSsig(2,:).' + ye/(2*area);
    E.Rx(:,i)=Rx; E.Ry(:,i)=Ry;

    H = chunkerkerneval(chnkr, KH, sigma, targ);         % (4Ne×1)
    E.Rxx(:,i) = 1/(2*area) + H(1:Ne);
    E.Rxy(:,i) = H((Ne+1):2*Ne);
    E.Ryx(:,i) = H((2*Ne+1):3*Ne);
    E.Ryy(:,i) = 1/(2*area) + H((3*Ne+1):4*Ne);

    DX = xe - xs(i);  DY = ye - ys(i);  R2 = DX.^2 + DY.^2;  c1 = -1/(2*pi);
    dSx = c1 * DX ./ R2;   dSy = c1 * DY ./ R2;
    E.Gx(:,i) = Rx + dSx;
    E.Gy(:,i) = Ry + dSy;

    Hs = hesslap_s_eval(struct('r',[xs(i);ys(i)]), targ);
    E.Gxx(:,i) = E.Rxx(:,i) + Hs(1:Ne);
    E.Gxy(:,i) = E.Rxy(:,i) + Hs((Ne+1):2*Ne);
    E.Gyx(:,i) = E.Ryx(:,i) + Hs((2*Ne+1):3*Ne);
    E.Gyy(:,i) = E.Ryy(:,i) + Hs((3*Ne+1):4*Ne);
  end
end

% Put R on the diagonal, not G
coinc = @(xe,ye,xs,ys) (abs(xe-xs)<1e-15) & (abs(ye-ys)<1e-15);
for i=1:Ns
  mask = coinc(xe,ye,xs(i),ys(i));
  if any(mask)
    E.G(mask,i) = E.R(mask,i);
    if want_derivs
      E.Gx(mask,i)=E.Rx(mask,i); E.Gy(mask,i)=E.Ry(mask,i);
      E.Gxx(mask,i)=E.Rxx(mask,i); E.Gxy(mask,i)=E.Rxy(mask,i);
      E.Gyx(mask,i)=E.Ryx(mask,i); E.Gyy(mask,i)=E.Ryy(mask,i);
    end
  end
end
end

function out = hesslap_s_eval(s, t)
xs=s.r(1,:); ys=s.r(2,:); xt=t.r(1,:); yt=t.r(2,:);
DX=xt.'-xs; DY=yt.'-ys; R2=DX.^2+DY.^2; R4=R2.^2; c=-1/(2*pi);
Hxx = c*( 1./R2 - 2*DX.^2 ./ R4 );
Hyy = c*( 1./R2 - 2*DY.^2 ./ R4 );
Hxy = c*( -2*DX.*DY ./ R4 );
out = [Hxx; Hxy; Hxy; Hyy];
end

function pack = build_sigma_for_source(xi)
    CHNK = CHNK_get();
    chnkr=CHNK.chnkr; vbn=CHNK.vbn; w=CHNK.w; area=CHNK.area;
    targs_bdry=CHNK.targs_bdry; alpha=CHNK.alpha;

    Sprime_src = chnk.lap2d.kern(struct('r',xi), targs_bdry, 'sprime');
    Ssrc_on_bd = chnk.lap2d.kern(struct('r',xi), targs_bdry, 's');

    rhs = -(-alpha)*Sprime_src - (vbn.' / area);
    t1  = alpha * 0.25 * (xi(1)^2 + xi(2)^2);
    t2  = -alpha * ( vbn * ( Ssrc_on_bd .* w ) );

    b        = rhs - ones(chnkr.npt,1)*(CHNK.base_s0 + CHNK.base_s1 + t1 + t2);
    sigma    = CHNK.Adec \ b;   % single backsolve per source
    pack.xi  = xi(:);
    pack.sigma = sigma;
end

% ---------- Evaluate G for arbitrary x using prebuilt sigma ----------
function G = eval_G_from_sigma(x, pack)
    CHNK = CHNK_get();
    chnkr=CHNK.chnkr; area=CHNK.area; alpha=CHNK.alpha;

    x = x(:);
    Sk   = kernel('laplace','s');
    Ssig = chunkerkerneval(chnkr, Sk, pack.sigma, struct('r',x));
    Ssrc = chnk.lap2d.kern(struct('r',pack.xi), struct('r',x), 's');
    poly = (x(1)^2 + x(2)^2)/(4*area);
    G = Ssig + (-alpha)*Ssrc + poly;
end

function pen = boundary_penalty(Cc, aa, bb, ph, margin, polyX, polyY)
    N = size(Cc,1);
    pen = 0;
    M = 60; th = linspace(0,2*pi,M+1); th(end)=[];

    clr = local_clearance_to_polygon(Cc, polyX, polyY);  % N×1

    for k=1:N
        ct = cos(ph(k)); st = sin(ph(k));
        XY = [aa(k)*cos(th); bb(k)*sin(th)];
        xy = [ct -st; st ct]*XY;
        xb = Cc(k,1)+xy(1,:); yb = Cc(k,2)+xy(2,:);
        ok = inpolygon(xb, yb, polyX, polyY);
        if any(~ok)
            pen = pen + sum(1 + 0.1*sqrt((xb(~ok)-Cc(k,1)).^2 + (yb(~ok)-Cc(k,2)).^2));
        end
        need = max(aa(k), bb(k)) + margin;
        if clr(k) < need
            pen = pen + (need - clr(k));
        end
    end
end

function pen = overlap_penalty(Cc, aa, bb, sep_margin)
    N = size(Cc,1);
    pen = 0;
    for i=1:N-1
        for j=i+1:N
            dij  = hypot(Cc(i,1)-Cc(j,1), Cc(i,2)-Cc(j,2));
            r_i  = 0.5*(aa(i)+bb(i));   % crude equivalent radius
            r_j  = 0.5*(aa(j)+bb(j));
            need = r_i + r_j + sep_margin;
            if dij < need, pen = pen + (need - dij); end
        end
    end
end

