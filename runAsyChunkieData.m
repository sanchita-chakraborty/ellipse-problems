o = asyChunkIE_onDataCell(21,'plot');

function o = asyChunkIE_onDataCell(cellNo, mode)
% Solve asymptotics on a real data cell using ChunkIE Green's.
% - No geometry simplification
% - Uses SolveAsy(o) below (unchanged algebra)
% - ChunkIE-backed G with diagonal regular-part handling
% - Parallelized hotspots + LU reuse

% -------------------- user knobs / numerics --------------------
K_quad          = 16;        % chunkie local order
EPS_build       = 1e-3;     % chunkerfuncuni cparams.eps
NCH_MIN_EXT     = 32;       % min #chunks on exterior
CHUNKS_PER_EDGE = 64;        % ~chunks per polygon edge
TIK             = 1e-12;    % Tikhonov for Sp system
GAUGE_ALPHA     = -1;       % gauge constant
DO_PARFOR       = false;    % set true to enable parfor

% -------------------- args --------------------
if nargin<1, cellNo = 1; end
if nargin<2, mode = '';   end

% -------------------- chunkie path --------------------
projRoot = pwd;
addpath(fullfile(projRoot,'chunkie'),'-begin');
run(fullfile(projRoot,'chunkie','startup.m'));
assert(exist('chunkerfunc','file')>0 && exist('chunkerfuncuni','file')>0 ...
    && exist('chunkermat','file')>0, 'chunkIE not on path');

% -------------------- load data (no simplification) --------------------
S = load('fly_muscle_data.mat');
need = {'edgeX','edgeY','NucX','NucY','NucMajor','NucMinor','NucAngle','startPoints','endPoints'};
for k=1:numel(need), assert(isfield(S,need{k}), 'Missing "%s"', need{k}); end

% exterior polygon
X = S.edgeX(cellNo,:).'; Y = S.edgeY(cellNo,:).';
g = isfinite(X) & isfinite(Y);  X = X(g);  Y = Y(g);
[X,Y] = close_and_drop_dupes(X,Y);
assert(X(1)==X(end) && Y(1)==Y(end),'Failed to close cell boundary');

% nuclei (centers & ellipse params)
idx  = S.startPoints(cellNo):S.endPoints(cellNo);
nucX = S.NucX(idx);  nucY = S.NucY(idx);
aMaj = S.NucMajor(idx); bMin = S.NucMinor(idx); ang = S.NucAngle(idx);
gg   = isfinite(nucX) & isfinite(nucY) & isfinite(aMaj) & isfinite(bMin) & isfinite(ang);
nucX = nucX(gg); nucY = nucY(gg); aMaj=aMaj(gg); bMin=bMin(gg); ang=ang(gg);
if isempty(nucX), error('Cell %d has no finite nuclei.',cellNo); end
if strcmpi(mode,'first10') && numel(nucX)>10
    nucX=nucX(1:10); nucY=nucY(1:10); aMaj=aMaj(1:10); bMin=bMin(1:10); ang=ang(1:10);
end
N = numel(nucX);

% -------------------- chunkie build (uniform chunks) --------------------
pref    = struct('k',K_quad);
cparams = struct('eps',EPS_build,'nover',0,'maxchunklen',inf,'nchmax',1e8);
seg_ext = pwlin_unitparam(X,Y);
Nedges  = numel(X)-1;
nch_ext = max( max(NCH_MIN_EXT, CHUNKS_PER_EDGE*Nedges), 1 );
chnkr   = chunkerfuncuni(seg_ext, nch_ext, cparams, pref);

% area via boundary identity v=(x^2+y^2)/4
xx = chnkr.r(1,:); yy = chnkr.r(2,:);
nx = chnkr.n(1,:); ny = chnkr.n(2,:);
w  = chnkr.wts(:);
vbn  = (xx.*nx + yy.*ny)/2;
area = vbn(:).'*w;  assert(area>0,'Area must be positive');

% boundary operators for Neumann solve
Sk  = kernel('laplace','s');
Skp = kernel('laplace','sprime');
Smat= chunkermat(chnkr,Sk);
Kp  = chunkermat(chnkr,Skp);
Sp  = 0.5*eye(chnkr.npt) + Kp;

% rank-1 compatibility correction (precompute)
vb       = (xx.^2 + yy.^2)/4;
v3bn     = (xx.^3.*nx + yy.^3.*ny)/12;
base_s0  = (v3bn * w) / area;
base_s1  = ((vb .* vbn) * w) / area;
vvec     = vbn .* (w.');
corr     = ones(chnkr.npt,1) * (vvec*Smat);  % rank-1
targs_bd = struct('r', chnkr.r, 'd', chnkr.d);

% ------------ stash (CHNK) + one-time factorization for reuse ------------
A_sys  = Sp + corr + TIK*eye(chnkr.npt);          % fixed matrix
Adec   = decomposition(A_sys,'lu');               % reuse this everywhere

% ---- ellipse pack lives in CHNK so everyone can see the holes -----------
ELL = struct();
ELL.cx  = nucX(:).';
ELL.cy  = nucY(:).';
ELL.a   = (aMaj(:).'/2);
ELL.b   = (bMin(:).'/2);
ELL.phi = ang(:).';
ELL.N   = numel(ELL.cx);

CHNK = struct('chnkr',chnkr,'Sp',Sp,'Smat',Smat,'vbn',vbn,'w',w,'area',area, ...
              'xx',xx,'yy',yy,'nx',nx,'ny',ny,'base_s0',base_s0,'base_s1',base_s1, ...
              'corr',corr,'targs_bdry',targs_bd,'alpha',GAUGE_ALPHA,'tik',TIK, ...
              'A',A_sys, 'Adec',Adec, ...
              'polyX',X(:).', 'polyY',Y(:).', 'ell',ELL, ...
              'debug', false);

% Init persistent store locally
initCHNK_store(CHNK);
assignin('base','CHNK',CHNK);   % for debugging & fallback

% If using parfor, push path + CHNK to workers
if DO_PARFOR
    pool = gcp('nocreate'); if isempty(pool), pool = parpool; end
    addAttachedFiles(pool, { mfilename('fullpath') });
    f1 = parfevalOnAll(@addpath, 0, fullfile(projRoot,'chunkie')); wait(f1)
    f2 = parfevalOnAll(@run,     0, fullfile(projRoot,'chunkie','startup.m')); wait(f2)
    CHNK_light = CHNK; if isfield(CHNK_light,'Adec'), CHNK_light = rmfield(CHNK_light,'Adec'); end
    f3 = parfevalOnAll(@initCHNK_store, 0, CHNK_light); wait(f3)
end

% -------------------- pick scalar epsilon from geometry --------------------
Lx = range(xx);  Ly = range(yy);
Lref = max([Lx, Ly, sqrt(area)]);
eps_scalar = 1./max(Lref, eps);

% -------------------- build 'o' and call SolveAsy --------------------
o = struct();
o.Domain = 'Disk';
o.Nc     = N;
o.x      = nucX(:).';
o.y      = nucY(:).';
o.a      = (aMaj(:).'/2);
o.b      = (bMin(:).'/2);
o.phi    = ang(:).';
o.eps    = eps_scalar * ones(1,N);
o.area   = area;

% evaluation points (exclude ellipses)
if strcmpi(mode,'plot')
    pad=0.05; xmin=min(X); xmax=max(X); xr=xmax-xmin; xmin=xmin-pad*xr; xmax=xmax+pad*xr;
    ymin=min(Y); ymax=max(Y); yr=ymax-ymin; ymin=ymin-pad*yr; ymax=ymax+pad*yr;
    Nx=80; Ny=600;
    [xax,yax] = deal(linspace(xmin,xmax,Nx),linspace(ymin,ymax,Ny));
    [Xg,Yg]=meshgrid(xax,yax);
    % polygon mask
    BW = inpolygon(Xg,Yg,X,Y);
    % subtract ellipse holes
    E = CHNK.ell;
    for t = 1:E.N
        ct = cos(E.phi(t)); st = sin(E.phi(t));
        Xr =  ct*(Xg-E.cx(t)) + st*(Yg-E.cy(t));
        Yr = -st*(Xg-E.cx(t)) + ct*(Yg-E.cy(t));
        BW = BW & ((Xr./E.a(t)).^2 + (Yr./E.b(t)).^2 > 1);
    end
    o.x_eval = Xg(BW); o.y_eval = Yg(BW);
else
    o.x_eval = o.x(:);
    o.y_eval = o.y(:);
end

% solve asymptotics
o = SolveAsy(o, DO_PARFOR);

% ALWAYS expose u0, u2, u (ε^2-scaled total)
if ~isfield(o,'u_asy0') || isempty(o.u_asy0) || ~isfield(o,'u_asy2')
    [u0_here,u2_here] = eval_u_fields(o.x_eval, o.y_eval, o, DO_PARFOR);
    o.u_asy0 = u0_here; o.u_asy2 = u2_here;
end

bad_corr = ~isfinite(o.chi2) || any(~isfinite(o.S2));
if bad_corr
    o.warn_nan_correction = true;
    o.u0 = o.u_asy0;
    o.u2 = zeros(size(o.u_asy0));
    o.u  = o.u0;
else
    o.u0 = o.u_asy0;
    o.u2 = o.u_asy2;
    o.u  = o.u0 + (eps_scalar^2)*o.u2;
end
o.eps_scalar = eps_scalar;

% handy callable
o.eval_u = @(Xq,Yq) deal_eval(o, Xq, Yq, eps_scalar, DO_PARFOR);

assignin('base','asyOut',o);

% plots
if strcmpi(mode,'plot')
    U  = NaN(size(Xg));  U(BW)  = o.u;
    U0 = NaN(size(Xg));  U0(BW) = o.u0;
    U2 = NaN(size(Xg));  U2(BW) = o.u2;
    fig = figure('Color','w');
    subplot(1,3,1); hold on
    contourf(xax,yax,U,40,'LineStyle','none'); axis xy equal tight
    plot([X;X(1)],[Y;Y(1)],'k-','LineWidth',1.1);
    plot(o.x, o.y, 'rx','MarkerSize',8,'LineWidth',1.1);
    colorbar; title(sprintf('u (total), \\epsilon=%.3g',o.eps_scalar));

    subplot(1,3,2); hold on
    contourf(xax,yax,U0,40,'LineStyle','none'); axis xy equal tight
    plot([X;X(1)],[Y;Y(1)],'k-','LineWidth',1.1);
    plot(o.x, o.y, 'rx','MarkerSize',8,'LineWidth',1.1);
    colorbar; title('u_0');

    subplot(1,3,3); hold on
    contourf(xax,yax,(o.eps_scalar^2)*U2,40,'LineStyle','none'); axis xy equal tight
    plot([X;X(1)],[Y;Y(1)],'k-','LineWidth',1.1);
    plot(o.x, o.y, 'rx','MarkerSize',8,'LineWidth',1.1);
    colorbar; title('\epsilon^2 u_2');

    % ---------- SAVE FIGURE ----------
    fname = sprintf('cell%d_plot_eps%.3g', cellNo, o.eps_scalar);
    saveas(fig, [fname '.png']);            % PNG (default quality)
    exportgraphics(fig, [fname '.pdf'], 'ContentType','vector');  % high-quality vector PDF
    disp(['Saved plots as ', fname, '.png and .pdf']);
end

end % ================= END MAIN =================

% ===================== LOCAL: G = -(1/2π)log r + R, Data/ChunkIE =====================
function [G,GradG,HessG] = getGy(y, xi)
% y  : 1x2 (or Nx2) targets
% xi : 1x2 source (the trap center)
% Uses CHNK (cached in base via initCHNK_store) to get the REGULAR part R.
% Singular is added only when y ~= xi. At y==xi we return R only.

    if isvector(y), y = y(:).'; end
    N = size(y,1);

    % --- REGULAR PART from ChunkIE (R, ∂_y R, ∂^2_y R) ---
    [R, dRy, Hry] = R_chunkie(y, xi);   % sizes: [N,1], [N,2], [2,2,N]

    % --- SINGULAR PART (add only when y != xi) ---
    dx  = y(:,1) - xi(1);
    dy  = y(:,2) - xi(2);
    r2  = dx.^2 + dy.^2;

    % y==xi mask (use geometry scale from CHNK)
    CHNK = CHNK_get();                      %#ok<NASGU>  % fetch
    Lref = evalin('base','CHNK.area')^0.5;       % mild scale
    tol2 = (1e-12*max(Lref,1))^2;
    self = (r2 <= tol2);

    % Start with pure regular
    G      = R;
    GradG  = dRy;
    HessG  = Hry;

    % Add singular where not self
    if any(~self)
        j     = find(~self);
        r2j   = r2(j);
        dxj   = dx(j);  dyj = dy(j);

        invr2 = 1./r2j;  invr4 = invr2.^2;
        c  = 1/(2*pi);

        s      = -c*0.5*log(r2j);
        sx     =  c*dxj .* invr2;
        sy     =  c*dyj .* invr2;
        sxx    =  c*( (dxj.^2 - dyj.^2) .* invr4 );
        sxy    =  c*( 2*dxj.*dyj .* invr4 );
        syy    =  c*( (dyj.^2 - dxj.^2) .* invr4 );

        G(j)          = G(j) + s;
        GradG(j,1)    = GradG(j,1) + sx;
        GradG(j,2)    = GradG(j,2) + sy;
        HessG(1,1,j)  = squeeze(HessG(1,1,j)) + sxx;
        HessG(1,2,j)  = squeeze(HessG(1,2,j)) + sxy;
        HessG(2,1,j)  = squeeze(HessG(2,1,j)) + sxy;
        HessG(2,2,j)  = squeeze(HessG(2,2,j)) + syy;
    end
end

% ===================== LOCAL: regular part via ChunkIE once per xi =====================
function [R, dRy, Hry] = R_chunkie(Y, xi)
% Solve for the REGULAR Neumann Green function R(y;xi) on a general domain:
%   Δ_y R = 0 in Ω,        ∂R/∂n = +∂s/∂n on Γ,      ⟨R⟩_Γ = 0,
% where s(y;xi) = -(1/2π)log|y-xi|.  Then G = s + R.
% We use your cached system: A = (Sp + corr + tik*I), LU in CHNK.Adec.

    if isvector(Y), Y = Y(:).'; end
    CHNK = CHNK_get();
    chnkr = CHNK.chnkr;

    % --- Build Neumann data: b = -∂s/∂n on boundary (note sign so ∂G/∂n = 0) ---
    xx = CHNK.xx; yy = CHNK.yy; nx = CHNK.nx; ny = CHNK.ny;  % boundary nodes+normals
    dx = xx - xi(1);
    dy = yy - xi(2);
    r2 = dx.^2 + dy.^2;
    c  = 1/(2*pi);
    ds_dn = c*(dx.*nx + dy.*ny) ./ r2;      % ∂s/∂n  (on Γ)
    b = -ds_dn(:);                           % so that ∂R/∂n = +∂s/∂n

    % --- Solve single-layer density for R using your prefactored A ---
    % A μ = b, with A = Sp + corr + tik*I already LU-factorized in CHNK.Adec
    mu = CHNK.Adec \ b;

    % --- Evaluate R at targets with SLP ---
    targs = struct('r', Y.');                % no 'd' field
    Sk = kernel('laplace','s');
    R  = chunkerkerneval(chnkr, Sk, mu, targs);   % Nt x 1

    % --- Grad/Hess of R at targets ---
    % If you don’t have explicit derivative kernels available, use tiny FD in y.
    Lref = max(range(chnkr.r,2));
    h = 1e-5 * max(Lref,1);

    ex = [1,0]; ey = [0,1];
    targs_px = struct('r', (Y + h*ex).');    % y + h e_x
    targs_mx = struct('r', (Y - h*ex).');    % y - h e_x
    targs_py = struct('r', (Y + h*ey).');
    targs_my = struct('r', (Y - h*ey).');
    targs_pp = struct('r', (Y + h*ex + h*ey).');
    targs_pm = struct('r', (Y + h*ex - h*ey).');
    targs_mm = struct('r', (Y - h*ex - h*ey).');
    targs_mp = struct('r', (Y - h*ex + h*ey).');

    Rp  = chunkerkerneval(chnkr, Sk, mu, targs_px);
    Rm  = chunkerkerneval(chnkr, Sk, mu, targs_mx);
    Rpy = chunkerkerneval(chnkr, Sk, mu, targs_py);
    Rmy = chunkerkerneval(chnkr, Sk, mu, targs_my);

    Rpp = chunkerkerneval(chnkr, Sk, mu, targs_pp);
    Rpm = chunkerkerneval(chnkr, Sk, mu, targs_pm);
    Rmm = chunkerkerneval(chnkr, Sk, mu, targs_mm);
    Rmp = chunkerkerneval(chnkr, Sk, mu, targs_mp);

    dRx = (Rp - Rm) /(2*h);
    dRy = [dRx, (Rpy - Rmy)/(2*h)];         % Nt x 2

    Rxx = (Rp - 2*R + Rm)/(h*h);
    Ryy = (Rpy - 2*R + Rmy)/(h*h);
    Rxy = (Rmm + Rpp - Rpm - Rmp)/(4*h*h);

    Nt = size(Y,1);
    Hry = zeros(2,2,Nt);
    Hry(1,1,:) = reshape(Rxx,1,1,Nt);
    Hry(2,2,:) = reshape(Ryy,1,1,Nt);
    Hry(1,2,:) = reshape(Rxy,1,1,Nt);
    Hry(2,1,:) = reshape(Rxy,1,1,Nt);
end

% --- eval_u wrapper ---
function varargout = deal_eval(o,Xq,Yq,eps_scalar,DO_PARFOR)
[u0,u2] = eval_u_fields(Xq,Yq,o,DO_PARFOR);
u = u0 + (eps_scalar^2)*u2;
varargout = {u0,u2,u};
end


% ===================== evaluator (diagonal-regularized) =====================
function [u0,u2,u2_unscaled] = eval_u_fields(Xq, Yq, o, DO_PARFOR)
Xq = Xq(:); Yq = Yq(:); Nq = numel(Xq);
u0 = o.chi0*ones(Nq,1);
u2 = o.chi2*ones(Nq,1);

CHNK = CHNK_get();
Lref = max(range([CHNK.xx CHNK.yy],2)); Lref = max(Lref,1);
tol_same = 1e-8 * Lref;

u0_loc = zeros(Nq,1);
u2_loc = zeros(Nq,1);

if DO_PARFOR
    parfor j = 1:Nq
        xq = [Xq(j) Yq(j)];
        u0j = o.chi0; u2j = o.chi2;
        for i = 1:o.Nc
            dx = xq(1)-o.x(i); dy = xq(2)-o.y(i);
            if hypot(dx,dy) < tol_same
                [R,GradR,HessR] = getRy(xq,[o.x(i) o.y(i)]);
                u0j = u0j - 2*pi*o.S0(i)*o.nu(i)*R;
                u2j = u2j +   pi*o.S0(i)*o.nu(i)*trace(o.Q_mats(:,:,i)*HessR ) ...
                            + 2*pi*(o.b_vec(:,i)'*(o.M_mats(:,:,i)*GradR')) ...
                            - 2*pi*o.S2(i)*o.nu(i)*R;
            else
                [G,GradG,HessG] = getGy(xq,[o.x(i) o.y(i)]);
                u0j = u0j - 2*pi*o.S0(i)*o.nu(i)*G;
                u2j = u2j +   pi*o.S0(i)*o.nu(i)*trace(o.Q_mats(:,:,i)*HessG ) ...
                            + 2*pi*(o.b_vec(:,i)'*(o.M_mats(:,:,i)*GradG')) ...
                            - 2*pi*o.S2(i)*o.nu(i)*G;
            end
        end
        u0_loc(j)=u0j; u2_loc(j)=u2j;
    end
else
    for j = 1:Nq
        xq = [Xq(j) Yq(j)];
        for i = 1:o.Nc
            dx = xq(1)-o.x(i); dy = xq(2)-o.y(i);
            if hypot(dx,dy) < tol_same
                [R,GradR,HessR] = getRy(xq,[o.x(i) o.y(i)]);
                u0(j) = u0(j) - 2*pi*o.S0(i)*o.nu(i)*R;
                u2(j) = u2(j) +   pi*o.S0(i)*o.nu(i)*trace(o.Q_mats(:,:,i)*HessR ) ...
                                + 2*pi*(o.b_vec(:,i)'*(o.M_mats(:,:,i)*GradR')) ...
                                - 2*pi*o.S2(i)*o.nu(i)*R;
            else
                [G,GradG,HessG] = getGy(xq,[o.x(i) o.y(i)]);
                u0(j) = u0(j) - 2*pi*o.S0(i)*o.nu(i)*G;
                u2(j) = u2(j) +   pi*o.S0(i)*o.nu(i)*trace(o.Q_mats(:,:,i)*HessG ) ...
                                + 2*pi*(o.b_vec(:,i)'*(o.M_mats(:,:,i)*GradG')) ...
                                - 2*pi*o.S2(i)*o.nu(i)*G;
            end
        end
    end
    u0_loc=u0; u2_loc=u2;
end

u0 = u0_loc; u2 = u2_loc; u2_unscaled = u2;
end


% ========================== HELPERS ==========================
function [Xo,Yo] = close_and_drop_dupes(X,Y)
if iscolumn(X), X=X.'; end
if iscolumn(Y), Y=Y.'; end
if X(1)~=X(end) || Y(1)~=Y(end), X=[X X(1)]; Y=[Y Y(1)]; end
d = hypot(diff(X),diff(Y)); keep = [true, d>0];
Xo = X(keep).'; Yo = Y(keep).';
if Xo(1)~=Xo(end) || Yo(1)~=Yo(1), Xo=[Xo;Xo(1)]; Yo=[Yo;Yo(1)]; end
end

function seg = pwlin_unitparam(X,Y)
P = [X(:) Y(:)]; if any(P(1,:)~=P(end,:)), P=[P; P(1,:)]; end
N = size(P,1)-1; L = N;
    function [r,dr,d2r] = fcurve(t)
        t=t(:).'; st = mod(t,2*pi)/(2*pi)*L; st(st==L)=0;
        k=max(1,min(N,floor(st)+1)); a=st-(k-1);
        p0=P(k ,:).'; p1=P(k+1,:).';
        r  = p0 + bsxfun(@times,(p1-p0),a);
        v  = (p1-p0); speed=L/(2*pi);
        dr = v*speed; d2r = zeros(2,numel(t));
    end
seg = @fcurve;
end

% Back-compat wrapper
function varargout = getRx(x,xi), [varargout{1:nargout}] = getRy(x,xi); end


% ===================== Internals (ChunkIE eval + stencils) =====================
% (Kept for reference; basefun path above is used by stencils)
function G = eval_G_chunkie(x, xi)
CHNK = CHNK_get();
chnkr=CHNK.chnkr; vbn=CHNK.vbn; w=CHNK.w; area=CHNK.area;
targs_bdry=CHNK.targs_bdry; alpha=CHNK.alpha;
if norm(x - xi) < 1e-14*max(range([CHNK.xx CHNK.yy],2))
    Greg = @(pt) eval_G_chunkie(pt, xi) + (1/(2*pi))*log(max(norm(pt - xi), eps));
    G = avg_dirs_4pt(Greg, xi); return
end
Sprime_src = chnk.lap2d.kern(struct('r',xi), targs_bdry, 'sprime');
Ssrc_on_bd = chnk.lap2d.kern(struct('r',xi), targs_bdry, 's');
rhs = -(-alpha)*Sprime_src - (vbn.' / area);
t1 = alpha * 0.25 * (xi(1)^2 + xi(2)^2);
t2 = -alpha * ( vbn * ( Ssrc_on_bd .* w ) );
b  = rhs - ones(chnkr.npt,1)*(CHNK.base_s0 + CHNK.base_s1 + t1 + t2);
sigma = CHNK.Adec \ b;
Sk = kernel('laplace','s');
Ssig = chunkerkerneval(chnkr, Sk, sigma, struct('r',x));
Ssrc = chnk.lap2d.kern(struct('r',xi), struct('r',x), 's');
poly = (x(1)^2 + x(2)^2)/(4*area);
G = Ssig + (-alpha)*Ssrc + poly;
end

% step size control
function h = choose_h_base(x0)
    CHNK = CHNK_get();
    Lref = max(range([CHNK.xx CHNK.yy],2)); if ~isfinite(Lref)||Lref<=0, Lref=1; end
    h = max(CHNK.hfrac * Lref, 1e-6*Lref);
end

% ---------- domain predicates (polygon AND outside ellipses) --------------
function tf = is_inside_domain(pt)
    CHNK = CHNK_get();
    tf_poly = inpolygon(pt(1), pt(2), CHNK.polyX, CHNK.polyY);
    if ~tf_poly, tf = false; return; end
    tf = ~inside_any_ellipse(pt, CHNK.ell);
end

function tf = inside_any_ellipse(pt, ELL)
    x = pt(1); y = pt(2);
    c = cos(ELL.phi); s = sin(ELL.phi);
    dx = x - ELL.cx;  dy = y - ELL.cy;
    xr =  c.*dx + s.*dy;
    yr = -s.*dx + c.*dy;
    tf = any( (xr./ELL.a).^2 + (yr./ELL.b).^2 <= 1 );
end

% ---------- step from x0 to a point that lies in the physical domain -----
function pt = step_into_domain(x0, dir, h0)
    max_expand = 30; max_refine = 30;
    h = max(h0, eps);
    pt = x0 + h*dir; k = 0;
    while ~is_inside_domain(pt) && k < max_expand
        h = 2*h; pt = x0 + h*dir; k = k+1;
    end
    if ~is_inside_domain(pt)
        h = max(h0, eps); k = 0; dir = -dir;
        pt = x0 + h*dir;
        while ~is_inside_domain(pt) && k < max_expand
            h = 2*h; pt = x0 + h*dir; k = k+1;
        end
    end
    if is_inside_domain(pt)
        k = 0;
        while k < max_refine
            h_half = 0.5*h;
            pt_try = x0 + h_half*dir;
            if is_inside_domain(pt_try)
                pt = pt_try; h = h_half;
            else
                break
            end
            k = k+1;
        end
    end
end

function [fp,fm, hp,hm] = interior_pair(fun, x0, dir, h0)
    x0 = x0(:);
    xp = step_into_domain(x0,  dir, h0);
    xm = step_into_domain(x0, -dir, h0);
    hp = norm(xp - x0); hm = norm(xm - x0);
    fp = fun(xp); fm = fun(xm);
end

% ===================== Asymptotic solver (unchanged algebra; diag-safe) ===
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

% ===================== Persistent CHNK store (for parfor) ====================
function initCHNK_store(CHNK_in)
    persistent gCHNK gAdec
    gCHNK = CHNK_in;
    if ~isfield(gCHNK,'A') || isempty(gCHNK.A)
        error('initCHNK_store:MissingA','CHNK_in.A is required');
    end
    gAdec = decomposition(gCHNK.A,'lu');
end

function CHNK = CHNK_get()
%CHNK_GET  Retrieve worker-local CHNK; auto-hydrate from base if needed.
    persistent gCHNK gAdec
    if isempty(gCHNK)
        if evalin('base','exist(''CHNK'',''var'')==1')
            gCHNK = evalin('base','CHNK');
        else
            error('CHNK_get:Uninitialized', ...
                  'CHNK store not initialized on this worker and not found in base.');
        end
    end
    if isempty(gAdec)
        if ~isfield(gCHNK,'A') || isempty(gCHNK.A)
            error('CHNK_get:MissingA', 'CHNK.A is missing or empty; cannot factorize.');
        end
        gAdec = decomposition(gCHNK.A, 'lu');
    end
    CHNK = gCHNK; CHNK.Adec = gAdec;
end

% ---------- Build sigma once for a given source xi ----------
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