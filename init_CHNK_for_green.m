function init_CHNK_for_green(cellNo)
% Build CHNK cache for green_data/green_regular_only (no SolveAsy).
% Requires: chunkie on path, fly_muscle_data.mat present, and your
%           existing initCHNK_store / CHNK_get / build_sigma_for_source.

% --- numerics (match your defaults) ---
K_quad          = 16;
EPS_build       = 1e-3;
NCH_MIN_EXT     = 32;
CHUNKS_PER_EDGE = 32;
TIK             = 1e-12;
GAUGE_ALPHA     = -1;

% --- chunkie path sanity ---
projRoot = pwd;
addpath(fullfile(projRoot,'chunkie'),'-begin');
run(fullfile(projRoot,'chunkie','startup.m'));
assert(exist('chunkerfunc','file')>0 && exist('chunkerfuncuni','file')>0 ...
    && exist('chunkermat','file')>0, 'chunkie not on path');

% --- load data ---
S = load('fly_muscle_data.mat');
need = {'edgeX','edgeY','NucX','NucY','NucMajor','NucMinor','NucAngle','startPoints','endPoints'};
for k=1:numel(need), assert(isfield(S,need{k}), 'Missing "%s"', need{k}); end

X = S.edgeX(cellNo,:).'; Y = S.edgeY(cellNo,:).';
g = isfinite(X) & isfinite(Y); X = X(g); Y = Y(g);
if X(1)~=X(end) || Y(1)~=Y(end), X=[X;X(1)]; Y=[Y;Y(1)]; end

idx  = S.startPoints(cellNo):S.endPoints(cellNo);
cx = S.NucX(idx); cy = S.NucY(idx);
aMaj = S.NucMajor(idx); bMin = S.NucMinor(idx); ang = S.NucAngle(idx);
gg = isfinite(cx)&isfinite(cy)&isfinite(aMaj)&isfinite(bMin)&isfinite(ang);
cx=cx(gg); cy=cy(gg); aMaj=aMaj(gg); bMin=bMin(gg); ang=ang(gg);
assert(~isempty(cx),'Cell %d has no finite nuclei',cellNo);

% --- build chunker (uniform chunks) ---
pref    = struct('k',K_quad);
cparams = struct('eps',EPS_build,'nover',0,'maxchunklen',inf,'nchmax',1e8);
seg_ext = pwlin_unitparam_local(X,Y);
Nedges  = numel(X)-1;
nch_ext = max( max(NCH_MIN_EXT, CHUNKS_PER_EDGE*Nedges), 1 );
chnkr   = chunkerfuncuni(seg_ext, nch_ext, cparams, pref);

xx = chnkr.r(1,:); yy = chnkr.r(2,:);
nx = chnkr.n(1,:); ny = chnkr.n(2,:); w = chnkr.wts(:);

% area via boundary identity
vbn  = (xx.*nx + yy.*ny)/2;
area = vbn(:).'*w;  assert(area>0,'Area must be positive');

% --- boundary operators (Neumann system for R) ---
Sk  = kernel('laplace','s');     Smat = chunkermat(chnkr,Sk);
Skp = kernel('laplace','sprime');Kp   = chunkermat(chnkr,Skp);
Sp  = 0.5*eye(chnkr.npt) + Kp;

% rank-1 compatibility bits
vb       = (xx.^2 + yy.^2)/4;
v3bn     = (xx.^3.*nx + yy.^3.*ny)/12;
base_s0  = (v3bn * w) / area;
base_s1  = ((vb .* vbn) * w) / area;
vvec     = vbn .* (w.');
corr     = ones(chnkr.npt,1) * (vvec*Smat);
targs_bd = struct('r', chnkr.r, 'd', chnkr.d);

A_sys  = Sp + corr + TIK*eye(chnkr.npt);
Adec   = decomposition(A_sys,'lu');

% --- stash minimal CHNK (geometry + LU) ---
ELL = struct();
ELL.cx  = cx(:).';
ELL.cy  = cy(:).';
ELL.a   = (aMaj(:).'/2);
ELL.b   = (bMin(:).'/2);
ELL.phi = ang(:).';
ELL.N   = numel(ELL.cx);

CHNK = struct('chnkr',chnkr,'Sp',Sp,'Smat',Smat,'vbn',vbn,'w',w,'area',area, ...
              'xx',xx,'yy',yy,'nx',nx,'ny',ny,'base_s0',base_s0,'base_s1',base_s1, ...
              'corr',corr,'targs_bdry',targs_bd,'alpha',GAUGE_ALPHA,'tik',TIK, ...
              'A',A_sys, 'Adec',Adec, ...
              'polyX',X(:).', 'polyY',Y(:).', 'ell',ELL, ...
              'cellNo',cellNo, 'debug', false);

% >>> IMPORTANT ORDER FIX <<<
% 1) Initialize the persistent store so CHNK_get() works
initCHNK_store(CHNK);
% 2) Put CHNK into base so CHNK_get() can also find it there
assignin('base','CHNK',CHNK);

% --- build sig_list (now CHNK_get works inside build_sigma_for_source) ---
sig_list = cell(1,ELL.N);
for i=1:ELL.N
    pack = build_sigma_for_source([ELL.cx(i); ELL.cy(i)]);
    sig_list{i} = pack.sigma;
end
CHNK.sig_list = sig_list;

% Update base copy (and re-init store with the augmented struct)
assignin('base','CHNK',CHNK);
initCHNK_store(CHNK);

disp('Initialized CHNK in base with: chnkr, area, polyX/Y, ell, sig_list, A/Adec');
end

% --- local helper for polygon parametrization ---
function seg = pwlin_unitparam_local(X,Y)
P = [X(:) Y(:)]; if any(P(1,:)~=P(end,:)), P=[P; P(1,:)]; end
N = size(P,1)-1; L = N;
    function [r,dr,d2r] = fcurve(t)
        t=t(:).'; st = mod(t,2*pi)/(2*pi)*L; st(st==L)=0;
        k=max(1,min(N,floor(st)+1)); a=st-(k-1);
        p0=P(k ,:).'; p1=P(k+1,:).';
        r  = p0 + (p1-p0).*a;
        speed=L/(2*pi); v = (p1-p0)*speed;
        dr = v; d2r = zeros(2,numel(t));
    end
seg = @fcurve;
end


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
