%% Green's on DATA geometry (MULTI nuclei) — CRASH-PROOF BATCHED EVAL
clearvars; close all; clc;

% (Optional: reduce threading to avoid MKL/OpenMP flakiness in heavy MEX)
% maxNumCompThreads(1);  % uncomment if you’ve seen random crashes

% --- ChunkIE on path ---
projRoot = pwd;
addpath(fullfile(projRoot,'chunkie'),'-begin');
run(fullfile(projRoot,'chunkie','startup.m'));
assert(exist('chunkerfunc','file')>0 && exist('chunkermat','file')>0, 'chunkIE not on path');

% --- Load data geometry + nuclei ---
S = load('fly_muscle_data.mat');
cellNo = 1;
X = S.edgeX(cellNo,:).';  Y = S.edgeY(cellNo,:).';
g = isfinite(X)&isfinite(Y);  X=X(g); Y=Y(g);
if ~(X(1)==X(end) && Y(1)==Y(end)), X(end+1)=X(1); Y(end+1)=Y(1); end

idx  = S.startPoints(cellNo):S.endPoints(cellNo);
nucX = S.NucX(idx);  nucY = S.NucY(idx);
gg   = isfinite(nucX)&isfinite(nucY);  nucX=nucX(gg); nucY=nucY(gg);
Nc   = numel(nucX);  if Nc==0, error('No nuclei for this cell.'); end

% --- Build chunker from polygon (safer resolution to avoid RAM spikes) ---
pref = struct('k',16);                            % lower order = lighter
cparams = struct('eps',1e-6,'nover',0,'maxchunklen',0.8);
seg_ext = pwlin_unitparam(X,Y);
CHUNKS_PER_EDGE = 16;                              % gentler than 32
nch_ext = max(CHUNKS_PER_EDGE*(numel(X)-1), 64);
chnkr = chunkerfuncuni(seg_ext, nch_ext, cparams, pref);

% --- Kernels & boundary matrices ---
Sk  = kernel('laplace','s');
Skp = kernel('laplace','sprime');
Smat  = chunkermat(chnkr,Sk);
Spmat = 0.5*eye(chnkr.npt) + chunkermat(chnkr,Skp);

% --- Boundary scalars/identities ---
xx = chnkr.r(1,:); yy = chnkr.r(2,:);
nx = chnkr.n(1,:); ny = chnkr.n(2,:);
w  = chnkr.wts(:);

vb  = (xx.^2 + yy.^2)/4;
vbn = (xx.*nx + yy.*ny)/2;
area = vbn*w;                         if area<=0, error('Area must be >0'); end

v3bn = (xx.^3.*nx + yy.^3.*ny)/12;
s0 = v3bn*w/area;
s1 = (vb.*vbn)*w/area;

vvec = vbn .* (w.');
corr   = ones(chnkr.npt,1) * (vvec*Smat);
syscorr = Spmat + corr;
Adec = decomposition(syscorr,'lu');

alph = -1;

% --- Prefactor done. Build RHS/solve ONCE PER NUCLEUS (tiny) ---
SIG  = cell(1,Nc);
SRCS = cell(1,Nc);
for i = 1:Nc
    src = struct('r',[nucX(i); nucY(i)]);
    SRCS{i} = src;

    rhs = -(-alph * chnk.lap2d.kern(src, chnkr, 'sprime'));

    t1 = alph*(src.r(1)^2 + src.r(2)^2)/4;
    t2 = -alph*( vbn * ( chnk.lap2d.kern(src, chnkr, 's') .* w ) );

    rhstot = rhs - ones(chnkr.npt,1)*(s0 + s1 + t1 + t2);
    SIG{i} = Adec \ rhstot;
end

% --- Build plotting grid & INTERIOR MASK ONCE ---
pad = 0.05;
xmin=min(X); xmax=max(X); xr=xmax-xmin; xmin=xmin-pad*xr; xmax=xmax+pad*xr;
ymin=min(Y); ymax=max(Y); yr=ymax-ymin; ymin=ymin-pad*yr; ymax=ymax+pad*yr;

Nx=150; Ny=600;                                   % modest grid (adjust)
[xax,yax]=deal(linspace(xmin,xmax,Nx), linspace(ymin,ymax,Ny));
[Xg,Yg]=meshgrid(xax,yax);

% interior mask in batches (avoid huge single call)
in = false(size(Xg));
B = 80000;                                        % batch size (#targets)
XT = [Xg(:)'; Yg(:)'];
NT = numel(Xg);
for s = 1:B:NT
    t = min(NT, s+B-1);
    tt = struct('r', XT(:,s:t));
    in(s:t) = chunkerinterior(chnkr, tt);
end
in = reshape(in,size(Xg));

% --- Evaluate sum_i G(y;xi_i) in tiles to cap memory ---
Gsum = zeros(size(Xg));
for s = 1:B:NT
    t = min(NT, s+B-1);
    targ = struct('r', XT(:,s:t));

    % accumulate contributions for this tile
    f_tile = zeros(1, t-s+1);
    % singular polynomial |y|^2/(4*area) term (same for all i)
    xy2 = 0.25/area * (XT(1,s:t).^2 + XT(2,s:t).^2);

    for i = 1:Nc
        % SLP from regular density
        f_slp = chunkerkerneval(chnkr, Sk, SIG{i}, targ).';  % row
        % singular
        f_sing = (-alph) * chnk.lap2d.kern(SRCS{i}, targ, 's').';
        f_tile = f_tile + (f_slp + f_sing + xy2);
    end

    % write back (mask later)
    Gsum(s:t) = f_tile;
end
Gsum(~in) = NaN;
Gsum = reshape(Gsum, size(Xg));

% --- Quick look (PNG only, LaTeX-styled titles if you want) ---
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

fig = figure('Color','w','Visible','off');
imagesc([xmin xmax],[ymin ymax], Gsum);
set(gca,'YDir','normal'); axis image;
colormap(parula); colorbar;
title('Sum of $G(y;\xi_i)$ on data geometry');
hold on; plot([X;X(1)],[Y;Y(1)],'k-','LineWidth',1.1);
plot(nucX, nucY,'rx','MarkerSize',8,'LineWidth',1.1); hold off;

drawnow;
exportgraphics(fig, sprintf('greens_cell%d.png',cellNo), 'Resolution', 400, 'BackgroundColor','white');
close(fig);

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

