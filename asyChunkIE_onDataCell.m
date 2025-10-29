function o = asyChunkIE_onDataCell(cellNo, mode)
% Solve asymptotics on a real data cell using ChunkIE Green's (Neumann).
% - No geometry simplification
% - ChunkIE-backed G with diagonal regular-part handling (s + R split)
% - Reuses a single LU (decomposition) and one regular-part density per nucleus
% - PNG export; LaTeX text; Dirichlet on ellipses enforced visually (plot mode)
%
% Usage:
%   o = asyChunkIE_onDataCell()               % default cell 21, plot
%   o = asyChunkIE_onDataCell(21,'plot')      % plot fields on a grid
%   o = asyChunkIE_onDataCell(21,'eval')      % evaluate only at nuclei

% -------------------- user knobs / numerics --------------------
K_quad          = 16;        % chunkie local order (12 is snappy; use 16 for final)
EPS_build       = 1e-3;      % chunkerfuncuni cparams.eps
NCH_MIN_EXT     = 32;        % min #chunks on exterior
CHUNKS_PER_EDGE = 32;        % ~chunks per polygon edge
TIK             = 1e-12;     % Tikhonov for Sp system
GAUGE_ALPHA     = -1;        % gauge constant (alpha)
DO_PARFOR       = false;     % keep false (decomposition can't be serialized)
ENFORCE_SYM     = false;     % set true for final symmetry/reciprocity averaging
% -------------------- DEBUG SWITCHES --------------------
DEBUG   = true;
DBG_LVL = 1;        % 0: quiet, 1: key milestones, 2: heavy checks
set_dbg(DEBUG, DBG_LVL);

% -------------------- args --------------------
if nargin<1, cellNo = 21; end
if nargin<2, mode   = 'plot'; end

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
assert(~isempty(nucX), 'Cell %d has no finite nuclei.',cellNo);
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
Smat  = chunkermat(chnkr,Sk);
Kp    = chunkermat(chnkr,Skp);
Sp    = 0.5*eye(chnkr.npt) + Kp;

vb   = (xx.^2 + yy.^2)/4;                     % scalar test fn
vbn  = (xx.*nx + yy.*ny)/2;                   % normal · grad(v)
v3bn = (xx.^3.*nx + yy.^3.*ny)/12;            % for s0

% geometry scalars (same as in NeumR_2D_Bulk_Int)
base_s0 = (v3bn(:).'*w) / area;               % 1×1
base_s1 = ((vb(:).*vbn(:)).'*w) / area;       % 1×1

s0    = ((xx.^3.*nx + yy.^3.*ny)/12 * w) / area;
s1    = ((vb(:).*vbn(:))' * w)      / area;

vvec    = vbn .* (w.');                      % 1×npt
corrmat = ones(chnkr.npt,1) * (vvec * Smat);
A_sys   = Sp + corrmat;                      % no /area, no ones*w'
A_sys   = A_sys + TIK*eye(chnkr.npt);
Adec    = decomposition(A_sys,'lu');

% ---- boundary targets for source kernels (s, sprime, etc) ----
targs_bdry = struct('r', chnkr.r);
if isfield(chnkr,'d')
    targs_bdry.d = chnkr.d;
end

% ---- ellipse pack lives in CHNK so everyone can see the holes -----------
ELL = struct();
ELL.cx  = nucX(:).';
ELL.cy  = nucY(:).';
ELL.a   = (aMaj(:).'/2);
ELL.b   = (bMin(:).'/2);
ELL.phi = ang(:).';
ELL.N   = numel(ELL.cx);

% stash in CHNK (update base_s0/base_s1 too)
CHNK = struct('chnkr',chnkr,'Sp',Sp,'Smat',Smat,'vbn',vbn,'w',w,'area',area, ...
              'xx',xx,'yy',yy,'nx',nx,'ny',ny,'base_s0',base_s0,'base_s1',base_s1, ...
              'corr',corrmat,'targs_bdry',targs_bdry,'alpha',GAUGE_ALPHA,'tik',TIK, ...
              'A',A_sys,'Adec',Adec, ...
              'polyX',X(:).','polyY',Y(:).','ell',ELL, ...
              'debug', false);

dbg(1,'[CHNK] npt=%d  area=%.6g  eps_build=%.1e  k=%d  nch=%d\n', ...
    chnkr.npt, area, EPS_build, K_quad, chnkr.nch);



% ========================================================================
% === Build regular-part densities for ALL nuclei (single solve each) ====
alpha = GAUGE_ALPHA;
vb_all  = (xx.^2 + yy.^2)/4;
vbn_all = (xx.*nx + yy.*ny)/2;
area_all= CHNK.area;

s0 = ( (xx.^3.*nx + yy.^3.*ny)/12 * w ) / area_all;     % equals base_s0
s1 = ( (vb_all(:).*vbn_all(:)).' * w ) / area_all;      % equals base_s1

CHNK.src_list = cell(1,ELL.N);
CHNK.sig_list = cell(1,ELL.N);
for i = 1:ELL.N
    src = struct('r',[ELL.cx(i); ELL.cy(i)]);
    CHNK.src_list{i} = src;

    rhs  = -(-alpha) * chnk.lap2d.kern(src, chnkr, 'sprime');
    t1   = alpha * (src.r(1)^2 + src.r(2)^2) / 4;
    t2   = -alpha * ( vbn(:).' * ( chnk.lap2d.kern(src, chnkr, 's') .* w ) );
    
    rhstot = rhs ...
           - (vbn.' / area) ...
           - ones(chnkr.npt,1) * ((s0 + s1 + t1 + t2) / area);

    CHNK.sig_list{i} = Adec \ rhstot;

    res_i = (A_sys * CHNK.sig_list{i}) - rhstot;
    dbg(2,'[SIG] i=%d  ||A*sigma-rhs||_2=%.3e  max|sigma|=%.3e\n', ...
        i, norm(res_i), max(abs(CHNK.sig_list{i})));
end
dbg(1,'[SIG] built %d regular-part densities\n', ELL.N);

% --- Build packs ONCE so the dataset never rebuilds sigma ---
CHNK.packList = cell(1, ELL.N);
for i = 1:ELL.N
    CHNK.packList{i} = struct('xi',[ELL.cx(i);ELL.cy(i)], 'sigma',CHNK.sig_list{i});
end

% refresh store with densities + packs
initCHNK_store(CHNK);
assignin('base','CHNK',CHNK);
dbg(1,'[STORE] CHNK stash ok; alpha=%g  tik=%g\n', CHNK.alpha, CHNK.tik);
% ========================================================================

% -------------------- pick scalar epsilon from geometry --------------------
Lx = range(xx);  Ly = range(yy);
Lref = max([Lx, Ly, sqrt(area)]);
eps_scalar = 1./max(Lref, eps);

% -------------------- build 'o' and call SolveAsy --------------------
o = struct();
o.Domain = 'Data';
o.Nc     = N;
o.x      = nucX(:).';
o.y      = nucY(:).';
o.a      = (aMaj(:).'/2);
o.b      = (bMin(:).'/2);
o.phi    = ang(:).';
o.eps    = eps_scalar * ones(1,N);
o.area   = area;

% evaluation points (exclude ellipses) or only at nuclei
if strcmpi(mode,'plot')
    pad=0.05; xmin=min(X); xmax=max(X); xr=xmax-xmin; xmin=xmin-pad*xr; xmax=xmax+pad*xr;
    ymin=min(Y); ymax=max(Y); yr=ymax-ymin; ymin=ymin-pad*yr; ymax=ymax+pad*yr;
    Nx=180; Ny=800;                                
    [xax,yax] = deal(linspace(xmin,xmax,Nx),linspace(ymin,ymax,Ny));
    [Xg,Yg]=meshgrid(xax,yax);
    BW = inpolygon(Xg,Yg,X,Y);
    E = CHNK.ell;
    for t = 1:E.N
        ct = cos(E.phi(t)); st = sin(E.phi(t));
        Xr =  ct*(Xg-E.cx(t)) + st*(Yg-E.cy(t));
        Yr = -st*(Xg-E.cx(t)) + ct*(Yg-E.cy(t));
        BW = BW & ((Xr./E.a(t)).^2 + (Yr./E.b(t)).^2 > 1);
    end
    o.x_eval = Xg(BW); o.y_eval = Yg(BW);
else
    o.x_eval = o.x(:); o.y_eval = o.y(:);
end

% ----- solve asymptotics (uses local dataset that reuses sigma packs) ----
o = SolveAsy(o, DO_PARFOR, ENFORCE_SYM);

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

% ---------- PLOTS: enforce Dirichlet(ellipses)=0 in the image ----------
if strcmpi(mode,'plot')
    BWcell = inpolygon(Xg, Yg, X, Y);
    E = CHNK.ell;
    BWell = false(size(Xg));
    for t = 1:E.N
        ct = cos(E.phi(t)); st = sin(E.phi(t));
        Xr =  ct*(Xg - E.cx(t)) + st*(Yg - E.cy(t));
        Yr = -st*(Xg - E.cx(t)) + ct*(Yg - E.cy(t));
        BWell = BWell | ((Xr./E.a(t)).^2 + (Yr./E.b(t)).^2 <= 1);
    end
    BWvalid = BWcell & ~BWell;

    Uimg  = nan(size(Xg));  Uimg(BWvalid)  = o.u;
    U0img = nan(size(Xg));  U0img(BWvalid) = o.u0;
    U2img = nan(size(Xg));  U2img(BWvalid) = o.u2;

    Uimg(BWell)  = 0;  U0img(BWell) = 0;  U2img(BWell) = 0;

    umax = max(Uimg(:),[],'omitnan'); levels_main = linspace(0,umax,40);

    set(groot,'defaultTextInterpreter','latex');
    set(groot,'defaultAxesTickLabelInterpreter','latex');
    set(groot,'defaultLegendInterpreter','latex');

    fig = figure('Color','w','Visible','off');
    set(fig,'Renderer','painters');
    tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');

    tt = linspace(0,2*pi,200);

    nexttile; hold on
    contourf(xax,yax,Uimg,levels_main,'LineStyle','none'); axis xy equal tight
    plot([X;X(1)],[Y;Y(1)],'k-','LineWidth',1.1);
    for t = 1:E.N
        R = [cos(E.phi(t)) -sin(E.phi(t)); sin(E.phi(t)) cos(E.phi(t))];
        el = R*[E.a(t)*cos(tt); E.b(t)*sin(tt)];
        plot(E.cx(t)+el(1,:), E.cy(t)+el(2,:), 'k-', 'LineWidth',0.8);
        plot(o.x(t),o.y(t),'rx','MarkerSize',8,'LineWidth',1.1);
    end
    colorbar; caxis([0,umax]);
    title(sprintf('$u$ (total), $\\epsilon=%.5g$',o.eps_scalar),'Interpreter','latex');

    nexttile; hold on
    contourf(xax,yax,U0img,levels_main,'LineStyle','none'); axis xy equal tight
    plot([X;X(1)],[Y;Y(1)],'k-','LineWidth',1.1);
    for t = 1:E.N
        R = [cos(E.phi(t)) -sin(E.phi(t)); sin(E.phi(t)) cos(E.phi(t))];
        el = R*[E.a(t)*cos(tt); E.b(t)*sin(tt)];
        plot(E.cx(t)+el(1,:), E.cy(t)+el(2,:), 'k-', 'LineWidth',0.8);
        plot(o.x(t),o.y(t),'rx','MarkerSize',8,'LineWidth',1.1);
    end
    colorbar; caxis([0,umax]);
    title('$u_0$','Interpreter','latex');

    nexttile; hold on
    U2scaled = (o.eps_scalar^2)*U2img;
    umax2 = max(U2scaled(:),[],'omitnan');
    contourf(xax,yax,U2scaled,linspace(0,max(umax2,eps),40),'LineStyle','none');
    axis xy equal tight
    plot([X;X(1)],[Y;Y(1)],'k-','LineWidth',1.1);
    for t = 1:E.N
        R = [cos(E.phi(t)) -sin(E.phi(t)); sin(E.phi(t)) cos(E.phi(t))];
        el = R*[E.a(t)*cos(tt); E.b(t)*sin(tt)];
        plot(E.cx(t)+el(1,:), E.cy(t)+el(2,:), 'k-', 'LineWidth',0.8);
        plot(o.x(t),o.y(t),'rx','MarkerSize',8,'LineWidth',1.1);
    end
    colorbar; caxis([0,max(umax2,eps)]);
    title('$\epsilon^2 u_2$','Interpreter','latex');

    drawnow;
    fname = sprintf('cell%d_plot_eps%.5g',cellNo,o.eps_scalar);
    set(fig,'Units','inches','Position',[1 1 6.5 3.2],'PaperPositionMode','auto');
    exportgraphics(fig,[fname '.png'],'Resolution',600,'BackgroundColor','white');
    close(fig);

    disp(['Saved plot as ', fname, '.png (600 DPI, LaTeX formatting)']);
end
end  % ================= END MAIN =================


% --- eval_u wrapper ---
function varargout = deal_eval(o,Xq,Yq,eps_scalar,DO_PARFOR)
[u0,u2] = eval_u_fields(Xq,Yq,o,DO_PARFOR);
u = u0 + (eps_scalar^2)*u2;
varargout = {u0,u2,u};
end


% ===================== evaluator (diagonal-regularized) =====================
function [G,GradG,HessG] = getGy(Y, xi)
    [G,GradG,HessG] = green_data(Y, xi, 'y');
end
function [G,GradG,HessG] = getGx(X, xi)
    [G,GradG,HessG] = green_data(X, xi, 'x');
end
function [R,GradR,HessR] = getRy(x, y)
    [R,GradR,HessR] = green_regular_only(x, y, 'y');
end
function [R,GradR,HessR] = getRx(x, y)
    [R,GradR,HessR] = green_regular_only(x, y, 'x');
end

function [G,GradG,HessG] = green_data(T, xi, ~)
%GREEN_DATA  Local Neumann Green’s eval on data geometry at target(s) T.
% Uses prebuilt sigma for the nearest nucleus to source xi, adds the
% correct polynomial + constant shift for the regular part, then adds
% the singular piece and its derivatives (dropped at self-hit).
%
% Inputs:
%   T  : Nt x 2 array of target points
%   xi : 1 x 2 source location (selects cached density by nearest nucleus)
% Output:
%   G      : Nt x 1 values
%   GradG  : Nt x 2 gradients w.r.t. target
%   HessG  : 2 x 2 x Nt Hessians w.r.t. target

if isvector(T), T = T(:).'; end
Nt = size(T,1);

CHNK  = CHNK_get();
chnkr = CHNK.chnkr;
area  = CHNK.area;

% kernels
Sk    = kernel('laplace','s');
Kgrad = kernel.lap2d('sgrad');
KH    = kernel(); KH.eval = @hesslap_s_eval; KH.opdims=[4 1]; KH.sing='smooth';

% pick nearest cached sigma to xi
[~,i0] = min( (CHNK.ell.cx(:)-xi(1)).^2 + (CHNK.ell.cy(:)-xi(2)).^2 );
sig    = CHNK.sig_list{i0};
wts    = chnkr.wts(:);
wbar   = sum(sig .* wts);                    % constant shift in R

% targets
x = T(:,1); y = T(:,2);
targ = struct('r',[x.'; y.']);

% ---- Regular part R ----
Ssig  = chunkerkerneval(chnkr, Sk, sigma, targ);      % Nt×1
R     = Ssig + (x.^2 + y.^2)/(4*area) + wbar;         % add w̄

% ---- Full G = R + singular ----
DX   = x - xi(1);   DY = y - xi(2);   r2 = DX.^2 + DY.^2;
Gsing= -(1/(2*pi)) * log( max(sqrt(r2), eps) );
G    = R + Gsing;

% ---- Derivatives of the regular part ----
J   = chunkerkerneval(chnkr, Kgrad, sig, targ);    % 2*Nt x 1
JJ  = reshape(J,2,[]);
Rx  = JJ(1,:).' + x/(2*area);
Ry  = JJ(2,:).' + y/(2*area);

H   = chunkerkerneval(chnkr, KH, sig, targ);       % 4*Nt x 1
Rxx = 1/(2*area) + H(1:Nt);
Rxy =              H((Nt+1):2*Nt);
Ryx =              H((2*Nt+1):3*Nt);
Ryy = 1/(2*area) + H((3*Nt+1):4*Nt);

% ---- Singular derivatives (for G) ----
R2  = max(r2, eps);
c1  = -(1/(2*pi));

dSx = c1 * (DX ./ R2);
dSy = c1 * (DY ./ R2);

Hxx = c1 * ( 1./R2 - 2*DX.^2 ./ (R2.^2) );
Hyy = c1 * ( 1./R2 - 2*DY.^2 ./ (R2.^2) );
Hxy = c1 * ( -2*DX.*DY ./ (R2.^2) );

% assemble full derivatives of G = R + singular
Gx  = Rx + dSx;   Gy = Ry + dSy;

Gxx = Rxx + Hxx;
Gxy = Rxy + Hxy;
Gyx = Ryx + Hxy;     % symmetry Hyx = Hxy
Gyy = Ryy + Hyy;

% ---- Self-hit: drop singular pieces at x=xi ----
tol2 = (1e-12 * max(sqrt(area),1))^2;
self = (DX.^2 + DY.^2) <= tol2;
if any(self)
    G(self)  = R(self);
    Gx(self) = Rx(self);  Gy(self) = Ry(self);
    Gxx(self)= Rxx(self); Gxy(self)= Rxy(self);
    Gyx(self)= Ryx(self); Gyy(self)= Ryy(self);
end

% pack outputs
GradG = [Gx, Gy];
HessG = zeros(2,2,Nt);
HessG(1,1,:) = reshape(Gxx,1,1,Nt);
HessG(1,2,:) = reshape(Gxy,1,1,Nt);
HessG(2,1,:) = reshape(Gyx,1,1,Nt);
HessG(2,2,:) = reshape(Gyy,1,1,Nt);
end

function [R,GradR,HessR] = green_regular_only(x, y, diff_wrt)
    [R,GradR,HessR] = green_data(x, y, diff_wrt);
end

function [u0,u2,u2_unscaled] = eval_u_fields(Xq, Yq, o, DO_PARFOR)
Xq = Xq(:); Yq = Yq(:); Nq = numel(Xq);
u0 = o.chi0*ones(Nq,1);
u2 = o.chi2*ones(Nq,1);
mustfinite('[nu]',o.nu); mustfinite('[chi0]',o.chi0); mustfinite('[chi2]',o.chi2);

CHNK = CHNK_get();
Lref = max(range([CHNK.xx CHNK.yy],2)); Lref = max(Lref,1);
tol_same = 1e-8 * Lref;

u0_loc = zeros(Nq,1); u2_loc = zeros(Nq,1);

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
u0 = u0_loc; u2 = u2_loc; u2_unscaled = u2;

if ~isempty(Xq)
    j = round(numel(Xq)/2);
    dbg(2,'[EVAL] probe @ (%.3f,%.3f): u0=%.6g  u2=%.6g\n', Xq(j), Yq(j), u0(j), u2(j));
end
end


% ===================== Asymptotic solver (diag-safe, pack reuse) =========
function o = SolveAsy(o, DO_PARFOR, ENFORCE_SYM) %#ok<INUSD>
N = numel(o.x);
G_mat = zeros(N);  

% geometry-dependent small parameter
o.eps = o.eps(:).';  o.a = o.a(:).';  o.b = o.b(:).';
o.nu  = -1 ./ log( 0.5 .* o.eps .* (o.a + o.b) );

% ellipse geometric tensors
o.Q_mats = zeros(2,2,N);
o.M_mats = zeros(2,2,N);
for k = 1:N
    o.Q_mats(:,:,k) = -0.25*(o.a(k)^2 - o.b(k)^2) * ...
                      [ cos(2*o.phi(k))  sin(2*o.phi(k));
                        sin(2*o.phi(k)) -cos(2*o.phi(k)) ];
    o.M_mats(:,:,k) = -0.25*(o.a(k)+o.b(k))^2 * eye(2) + o.Q_mats(:,:,k);
end

% build Green dataset once (reusing sigmas)
CHNK = CHNK_get();
Esrc = NeumGR_dataset(o.x(:), o.y(:), o.x(:), o.y(:), true);

% diag & symmetry diagnostics
dself = abs(diag(Esrc.G) - diag(Esrc.R));
fprintf('[GR] max|G(ii)-R(ii)| = %.3e\n', max(dself));
symerr = max(max(abs(Esrc.G - Esrc.G.')));
fprintf('[GR] symmetry max|G-G''| = %.3e\n', symerr);

if ENFORCE_SYM
    Esrc.G   = 0.5*(Esrc.G + Esrc.G.');
    Tmp      = 0.5*(Esrc.Gx - Esrc.Gy.');
    Esrc.Gx  = Tmp; Esrc.Gy = -Tmp.';
    Esrc.Gxx = 0.5*(Esrc.Gxx + Esrc.Gyy.');
    Esrc.Gyy = Esrc.Gxx.';
    Tmp2     = 0.5*(Esrc.Gxy - Esrc.Gyx.');
    Esrc.Gxy = Tmp2; Esrc.Gyx = -Tmp2.';
end

% Pull self-regular parts
idx = (1:N).';
lin = sub2ind([N,N], idx, idx);
R0      = Esrc.R(lin).';
GR0     = [Esrc.Rx(lin).'; Esrc.Ry(lin).'];
HR0     = zeros(2,2,N);
HR0_xx  = Esrc.Rxx(lin).';  HR0_xy = Esrc.Rxy(lin).';
HR0_yx  = Esrc.Ryx(lin).';  HR0_yy = Esrc.Ryy(lin).';
for k=1:N, HR0(:,:,k) = [HR0_xx(k) HR0_xy(k); HR0_yx(k) HR0_yy(k)]; end

% Full Green matrix: off-diag = G, diag = R
G_mat = Esrc.G;
for k=1:N, G_mat(k,k) = R0(k); end
G_mat = (G_mat + G_mat.')/2;

% ------------------ Leading order system -------------------
M = [ (eye(N) + 2*pi*G_mat*diag(o.nu)), -ones(N,1); ...
       o.nu,                             0          ];
rhs0 = [zeros(N,1); o.area/(2*pi)];
A = M\rhs0;
o.S0   = A(1:N);
o.chi0 = A(end);

resL = M*A - rhs0;
fprintf('[LO] ||res||_2=%.3e  ||S0||_inf=%.3e  chi0=%.6g\n', ...
        norm(resL), norm(A(1:N),inf), A(end));
mustfinite('[S0]',o.S0);

% ------------------ Correction-term pieces -----------------
o.b_vec  = zeros(2,N);
o.H_mats = zeros(2,2,N);
for k = 1:N
    o.b_vec(:,k)    = -2*pi*o.S0(k)*o.nu(k)*GR0(:,k);
    o.H_mats(:,:,k) = -2*pi*o.S0(k)*o.nu(k)*HR0(:,:,k);
end

scaleSm = -2*pi * (o.S0(:)'.*o.nu(:)');
for k = 1:N
    gGx = Esrc.Gx(k,:);  gGy = Esrc.Gy(k,:);
    Hxx = Esrc.Gxx(k,:); Hxy = Esrc.Gxy(k,:);
    Hyx = Esrc.Gyx(k,:); Hyy = Esrc.Gyy(k,:);
    gGx(k)=0; gGy(k)=0; Hxx(k)=0; Hxy(k)=0; Hyx(k)=0; Hyy(k)=0;

    o.b_vec(1,k) = o.b_vec(1,k) + sum(scaleSm .* gGx);
    o.b_vec(2,k) = o.b_vec(2,k) + sum(scaleSm .* gGy);

    o.H_mats(1,1,k) = o.H_mats(1,1,k) + sum(scaleSm .* Hxx);
    o.H_mats(1,2,k) = o.H_mats(1,2,k) + sum(scaleSm .* Hxy);
    o.H_mats(2,1,k) = o.H_mats(2,1,k) + sum(scaleSm .* Hyx);
    o.H_mats(2,2,k) = o.H_mats(2,2,k) + sum(scaleSm .* Hyy);
end

% ------------------ RHS for correction ---------------------
rhs = zeros(N,1);
for k = 1:N
    GradXiR  = GR0(:,k).';   HessXiR = HR0(:,:,k);

    rhs_k = pi*o.S0(k)*o.nu(k)*trace(o.Q_mats(:,:,k)*HessXiR) ...
          + 2*pi*(o.b_vec(:,k)'*(o.M_mats(:,:,k)*GradXiR')) ...
          - (o.a(k)^2 + o.b(k)^2)/8 ...
          - 0.5*trace(o.Q_mats(:,:,k)*o.H_mats(:,:,k));

    gGx = Esrc.Gx(k,:);  gGy = Esrc.Gy(k,:);
    Hxx = Esrc.Gxx(k,:); Hxy = Esrc.Gxy(k,:);
    Hyx = Esrc.Gyx(k,:); Hyy = Esrc.Gyy(k,:);
    gGx(k)=0; gGy(k)=0; Hxx(k)=0; Hxy(k)=0; Hyx(k)=0; Hyy(k)=0;

    qH = zeros(1,N);
    for m=1:N
        if m==k, continue; end
        Hm = [Hxx(m) Hxy(m); Hyx(m) Hyy(m)];
        qH(m) = trace( o.Q_mats(:,:,m) * Hm );
    end
    rhs_k = rhs_k + pi * sum( (o.S0(:)'.*o.nu(:)').* qH );

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

resC = M*A - [rhs; 0];
fprintf('[C2] ||res||_2=%.3e  ||S2||_inf=%.3e  chi2=%.6g\n', ...
        norm(resC), norm(A(1:N),inf), A(end));
mustfinite('[S2]',o.S2);

% ------------------ Field on eval points -------------------
N_pts = numel(o.x_eval);
if N_pts>0
    CHNK = CHNK_get();
    Eev = NeumGR_dataset(o.x_eval(:), o.y_eval(:), o.x(:), o.y(:), true);
    o.u_asy0 = o.chi0*ones(N_pts,1) - 2*pi * ( Eev.G * (o.S0(:).*o.nu(:)) );

    % fix coincidences to use R instead of G
    for i=1:N
        mask = (abs(o.x_eval(:)-o.x(i))<1e-15) & (abs(o.y_eval(:)-o.y(i))<1e-15);
        if any(mask)
            o.u_asy0(mask) = o.u_asy0(mask) ...
                + 2*pi*o.S0(i)*o.nu(i) * ( Eev.G(mask,i) - Eev.R(mask,i) );
        end
    end

    o.u_asy2 = o.chi2*ones(N_pts,1);

    trQH = zeros(N_pts,N);
    for i=1:N
        trQH(:,i) = o.Q_mats(1,1,i)*Eev.Gxx(:,i) + ...
                    (o.Q_mats(1,2,i)+o.Q_mats(2,1,i))*Eev.Gxy(:,i) + ...
                     o.Q_mats(2,2,i)*Eev.Gyy(:,i);
    end
    term1 = pi * ( trQH * (o.S0(:).*o.nu(:)) );

    MGx = zeros(N_pts,N);  MGy = zeros(N_pts,N);
    for i=1:N
        MGx(:,i) = o.M_mats(1,1,i)*Eev.Gx(:,i) + o.M_mats(1,2,i)*Eev.Gy(:,i);
        MGy(:,i) = o.M_mats(2,1,i)*Eev.Gx(:,i) + o.M_mats(2,2,i)*Eev.Gy(:,i);
    end
    term2 = 2*pi * ( MGx * o.b_vec(1,:).' + MGy * o.b_vec(2,:).' );

    term3 = -2*pi * ( Eev.G * (o.S2(:).*o.nu(:)) );

    o.u_asy2 = o.u_asy2 + term1 + term2 + term3;
end

% --- boundary Neumann residual on u0 (sanity) ---
CHNK = CHNK_get(); chnkr = CHNK.chnkr;
xe = chnkr.r(1,:).'; ye = chnkr.r(2,:).';
nx = reshape(chnkr.n(1,:,:),[],1); ny = reshape(chnkr.n(2,:,:),[],1);
ne = [nx ny]; wt = chnkr.wts(:);
Ebd = NeumGR_dataset(xe, ye, o.x(:), o.y(:), true);
gradx = -2*pi * (Ebd.Gx * (o.S0(:).*o.nu(:)));
grady = -2*pi * (Ebd.Gy * (o.S0(:).*o.nu(:)));
flux  = gradx.*ne(:,1) + grady.*ne(:,2);
fprintf('[BC] mean(n·∇u0) = %.3e,  L2(n·∇u0) = %.3e\n', ...
        sum(flux.*wt)/sum(wt), sqrt(sum((flux.^2).*wt)/sum(wt)));
end


% ===================== NEUMANN GREEN DATASET (sigma reuse) ==============
function E = NeumGR_dataset(Xeval, Yeval, Xsrc, Ysrc, want_derivs)
%NEUMGR_DATASET  Neumann Green's function dataset on the data geometry.
% Builds matrices of R, G and their derivatives for targets (Xeval,Yeval)
% against sources (Xsrc,Ysrc), using prebuilt ChunkIE densities.
%
% Notes:
%  - Regular part R(x,y) = SLP[sigma_y](x) + |x|^2/(4*area) + wbar_y
%  - G(x,y) = R(x,y) -(1/2pi)log|x-y|
%  - Derivatives are w.r.t. the target x.
%  - wbar_y = sum_j sigma_y(j) * w_j (constant in x)
%
% Requires CHNK stash to be initialized (CHNK_get).

if nargin<5, want_derivs = true; end

CHNK  = CHNK_get();
chnkr = CHNK.chnkr;
area  = CHNK.area;

% Kernels
Sk    = kernel('laplace','s');
Kgrad = kernel.lap2d('sgrad');     % gradient wrt target x
KH    = kernel(); KH.eval = @hesslap_s_eval; KH.opdims=[4 1]; KH.sing='smooth';

% Shapes
xe = Xeval(:); ye = Yeval(:); Ne = numel(xe);
xs = Xsrc(:);  ys = Ysrc(:);  Ns = numel(xs);

% Targets container (for chunkerkerneval)
targ = struct('r', [xe.'; ye.']);

% Allocate outputs
E.R = zeros(Ne,Ns);
E.G = zeros(Ne,Ns);
if want_derivs
    Z = zeros(Ne,Ns);
    [E.Rx,E.Ry,E.Rxx,E.Rxy,E.Ryx,E.Ryy] = deal(Z,Z,Z,Z,Z,Z);
    [E.Gx,E.Gy,E.Gxx,E.Gxy,E.Gyx,E.Gyy] = deal(Z,Z,Z,Z,Z,Z);
end

% Loop over sources
for i = 1:Ns
    % ---- reuse cached sigma for the nearest prebuilt source ----
    % (Assumes CHNK.sig_list corresponds to nuclei at CHNK.ell.{cx,cy})
    [~,i0] = min( (CHNK.ell.cx(:)-xs(i)).^2 + (CHNK.ell.cy(:)-ys(i)).^2 );
    sigma  = CHNK.sig_list{i0};
    wbar   = sum(sigma .* chnkr.wts(:));  % constant shift in R

    % ----- Regular part R -----
    Ssig = chunkerkerneval(chnkr, Sk, sigma, targ);   % Ne x 1
    R    = Ssig + (xe.^2 + ye.^2)/(4*area) + wbar;    % add wbar
    E.R(:,i) = R;

    % ----- Full Green’s G = R + singular -----
    DX   = xe - xs(i);   DY = ye - ys(i);
    r2   = DX.^2 + DY.^2;
    Gsing= -(1/(2*pi)) * log( max(sqrt(r2), eps) );
    G    = R + Gsing;
    E.G(:,i) = G;

    if want_derivs
        % ---- gradient of R (regular part) ----
        J   = chunkerkerneval(chnkr, Kgrad, sigma, targ);  % 2*Ne x 1 stacked
        JJ  = reshape(J, 2, []);
        Rx  = JJ(1,:).' + xe/(2*area);
        Ry  = JJ(2,:).' + ye/(2*area);
        E.Rx(:,i) = Rx;  E.Ry(:,i) = Ry;

        % ---- Hessian of R (regular part) ----
        H    = chunkerkerneval(chnkr, KH, sigma, targ);    % (4*Ne) x 1
        E.Rxx(:,i) = 1/(2*area) + H(1:Ne);
        E.Rxy(:,i) =              H((Ne+1):2*Ne);
        E.Ryx(:,i) =              H((2*Ne+1):3*Ne);
        E.Ryy(:,i) = 1/(2*area) + H((3*Ne+1):4*Ne);

        % ---- singular derivatives (for G) ----
        R2  = max(r2, eps);
        c1  = -(1/(2*pi));

        dSx = c1 * (DX ./ R2);
        dSy = c1 * (DY ./ R2);

        Hxx = c1 * ( 1./R2 - 2*DX.^2 ./ (R2.^2) );
        Hyy = c1 * ( 1./R2 - 2*DY.^2 ./ (R2.^2) );
        Hxy = c1 * ( -2*DX.*DY ./ (R2.^2) );

        % G = R + singular => add singular pieces to R-derivatives
        E.Gx(:,i)  = Rx + dSx;
        E.Gy(:,i)  = Ry + dSy;
        E.Gxx(:,i) = E.Rxx(:,i) + Hxx;
        E.Gxy(:,i) = E.Rxy(:,i) + Hxy;
        E.Gyx(:,i) = E.Ryx(:,i) + Hxy;   % symmetry Hyx = Hxy
        E.Gyy(:,i) = E.Ryy(:,i) + Hyy;
    end

    % ---- self-hit consistency: drop singular at x=y ----
    self = (abs(xe - xs(i)) < 1e-14) & (abs(ye - ys(i)) < 1e-14);
    if any(self)
        E.G(self,i) = E.R(self,i);
        if want_derivs
            E.Gx(self,i)  = E.Rx(self,i);
            E.Gy(self,i)  = E.Ry(self,i);
            E.Gxx(self,i) = E.Rxx(self,i);
            E.Gxy(self,i) = E.Rxy(self,i);
            E.Gyx(self,i) = E.Ryx(self,i);
            E.Gyy(self,i) = E.Ryy(self,i);
        end
    end

    % ---- NaN hardening: micro-nudge along radial direction and recompute ----
    if want_derivs
        bad = ~isfinite(E.G(:,i)) | ~isfinite(E.Gx(:,i)) | ~isfinite(E.Gy(:,i)) | ...
              ~isfinite(E.Gxx(:,i))| ~isfinite(E.Gxy(:,i))| ~isfinite(E.Gyy(:,i));
    else
        bad = ~isfinite(E.G(:,i));
    end
    if any(bad)
        xe_bad = xe(bad); ye_bad = ye(bad);
        dx  = xe_bad - xs(i); dy = ye_bad - ys(i);
        Lref= max([range(xe(:)) range(ye(:)) sqrt(area) 1]);  % scalar and safe
        eta = 5e-13 * Lref; r = hypot(dx,dy); r(r==0)=1;
        xe2 = xe_bad + eta*(dx./r); ye2 = ye_bad + eta*(dy./r);
        targ2 = struct('r',[xe2.'; ye2.']);

        % R at nudged targets
        Ssig2 = chunkerkerneval(chnkr, Sk, sigma, targ2);
        R2v   = Ssig2 + (xe2.^2 + ye2.^2)/(4*area) + wbar;

        % G at nudged targets
        DX2 = xe2 - xs(i); DY2 = ye2 - ys(i); rr2 = DX2.^2 + DY2.^2;
        Gsing2 = -(1/(2*pi))*log( max(sqrt(rr2),eps) );
        G2     = R2v + Gsing2;

        E.G(bad,i) = G2;

        if want_derivs
            % gradients of R (nudged)
            J2   = chunkerkerneval(chnkr, Kgrad, sigma, targ2);
            JJ2  = reshape(J2,2,[]);
            Rx2  = JJ2(1,:).' + xe2/(2*area);
            Ry2  = JJ2(2,:).' + ye2/(2*area);

            % Hessian of R (nudged)
            H2   = chunkerkerneval(chnkr, KH, sigma, targ2);
            Rxx2 = 1/(2*area) + H2(1:numel(xe2));
            Rxy2 =              H2((numel(xe2)+1):2*numel(xe2));
            Ryx2 =              H2((2*numel(xe2)+1):3*numel(xe2));
            Ryy2 = 1/(2*area) + H2((3*numel(xe2)+1):4*numel(xe2));

            % singular derivatives at nudged points
            R2p  = max(rr2, eps);
            c1   = -(1/(2*pi));
            dSx2 = c1 * (DX2 ./ R2p);
            dSy2 = c1 * (DY2 ./ R2p);
            Hxx2 = c1 * ( 1./R2p - 2*DX2.^2 ./ (R2p.^2) );
            Hyy2 = c1 * ( 1./R2p - 2*DY2.^2 ./ (R2p.^2) );
            Hxy2 = c1 * ( -2*DX2.*DY2 ./ (R2p.^2) );

            E.Rx(bad,i)  = Rx2;                E.Ry(bad,i)  = Ry2;
            E.Rxx(bad,i) = Rxx2;               E.Rxy(bad,i) = Rxy2;
            E.Ryx(bad,i) = Ryx2;               E.Ryy(bad,i) = Ryy2;

            E.Gx(bad,i)  = Rx2 + dSx2;         E.Gy(bad,i)  = Ry2 + dSy2;
            E.Gxx(bad,i) = Rxx2 + Hxx2;        E.Gxy(bad,i) = Rxy2 + Hxy2;
            E.Gyx(bad,i) = Ryx2 + Hxy2;        E.Gyy(bad,i) = Ryy2 + Hyy2;
        end
    end
end
end

% ===================== Hessian of singular kernel =======================
function out = hesslap_s_eval(s, t)
    xs=s.r(1,:); ys=s.r(2,:); xt=t.r(1,:); yt=t.r(2,:);
    DX=xt.'-xs; DY=yt.'-ys; R2=DX.^2+DY.^2; R4=R2.^2; c=-(1/(2*pi));
    Hxx = c*( 1./R2 - 2*DX.^2 ./ R4 );
    Hyy = c*( 1./R2 - 2*DY.^2 ./ R4 );
    Hxy = c*( -2*DX.*DY ./ R4 );
    out = [Hxx; Hxy; Hxy; Hyy];
end


% ===================== Persistent CHNK store (no parfor) =================
function initCHNK_store(CHNK_in)
    persistent gCHNK gAdec
    gCHNK = CHNK_in;
    if ~isfield(gCHNK,'A') || isempty(gCHNK.A)
        error('initCHNK_store:MissingA','CHNK_in.A is required');
    end
    gAdec = decomposition(gCHNK.A,'lu'); %#ok<NASGU> keep local for CHNK_get
end

function CHNK = CHNK_get()
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


% ---------- Build sigma once for a given source xi (fallback) -----------
function pack = build_sigma_for_source(xi)
    CHNK = CHNK_get();
    chnkr=CHNK.chnkr; vbn=CHNK.vbn; w=CHNK.w; area=CHNK.area;
    targs_bdry=CHNK.targs_bdry; alpha=CHNK.alpha;

    Sprime_src = chnk.lap2d.kern(struct('r',xi), targs_bdry, 'sprime');
    Ssrc_on_bd = chnk.lap2d.kern(struct('r',xi), targs_bdry, 's');

    rhs = -(-alpha) * Sprime_src;
    t1  = alpha * 0.25 * (xi(1)^2 + xi(2)^2);
    t2  = -alpha * ( vbn(:).' * ( Ssrc_on_bd .* w ) );
    
    b = rhs ...
      - (vbn.' / area) ...
      - ones(chnkr.npt,1) * ( (CHNK.base_s0 + CHNK.base_s1 + t1 + t2) / area );
    
    sigma = CHNK.Adec \ b;

    pack.xi    = xi(:);
    pack.sigma = sigma;
end


% ========================== HELPERS =====================================
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

function mustfinite(name,A)
    if ~all(isfinite(A(:)))
        error('[FATAL] %s has NaN/Inf (min=%.3g max=%.3g)', name, min(A(:)), max(A(:)));
    end
end

function set_dbg(D,L)
  persistent D0 L0
  D0 = logical(D); L0 = double(L);
end

function dbg(lvl, varargin)
  persistent D0 L0
  if isempty(D0), D0 = false; L0 = 0; end
  if D0 && lvl <= L0
      fprintf(varargin{:});
  end
end
