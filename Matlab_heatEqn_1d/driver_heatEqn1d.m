%% problem to solve
% u_t - a*u_xx = b(x,t),  a > 0, x in [0,1], t in [0,T]
%      u(0,t)  = u(1,t) = 0,   t in [0,T]
%      u(x,0)  = sin(pi*x),    x in [0,1]
% with RHS b(x,t) = -sin(pi*x)(sin(t) - a*pi^2*cos(t))
%   => solution u(x,t) = sin(pi*x)*cos(t)
%
% discretization:
% ---------------
%  central FD in space
%  backward Euler in time
%   => (1/dt)*(u_{j,i} - u_{j,i-1}) - 
%          a/(dx^2)*(u_{j-1,i} - 2u_{j,i} + u_{j+1,i}) = b_{j,i}
%  <=> (1 + 2(a*dt/(dx^2)))u_{j,i} - (a*dt/(dx^2))(u_{j-1,i}+u_{j+1,i})
%         = u_{j,i-1} + dt*b_{j,i}
% for j = 0, ..., nx+1 and i = 1, ..., nt+1
% using the BCs, we have
%   u_{0,i} = u_{nx+1,i} = 0 for all i = 0,...,nt+1
% since these are homogeneous BCs, we only solve the system in the interior
% of the spatial domain, i.e., for j = 1, ..., nx.
% At each time step i = 1, ..., nt+1, we obtain the linear system
%   | 1+2ar   -ar                     | |  u_{1,i}   |   |  u_{1,i-1}   |
%   |  -ar   1+2ar  -ar               | |  u_{2,i}   |   |  u_{2,i-1}   |
%   |         ...   ...    ...        | |    ...     | = |     ...      |
%   |               -ar   1+2ar  -ar  | | u_{nx-1,i} |   | u_{nx-1,i-1} |
%   |                      -ar  1+2ar | |  u_{nx,i}  |   |  u_{nx,i-1}  |
%
%                                                        |  dt*b_{1,i}   |
%                                                        |  dt*b_{2,i}   |
%                                                        |     ...       |
%                                                        | dt*b_{nx-1,i} |
%                                                        |  dt*b_{nx,i}  |
% with r = (dt/dx^2), which we denote by 
%     Mu_i = u_{i-1} + dt*b_i.
% This leads to the time-stepping problem
%     u_i = M^{-1}(u_{i-1} + dt*b_i)
% which is implemented as time integrator function Phi
%     u_i = Phi(u_{i-1}, t_{i}, t_{i-1}, app)

%% plot options
options_plot   = {'LineWidth',3,'MarkerSize',14};
options_labels = {'FontSize',20};

%% problem parameters
a  = 1;                    % diffusion coefficient
T  = 5;                    % final time
nt = 100;                  % number of time steps
t  = linspace(0,T,nt+1)';  % time domain
dt = T/nt;                 % time-step size
X  = 1;                    % interval bound of spatial domain
nx = 16;                   % number of dofs in space
dx = X/nx;                 % spatial step size
x  = linspace(0,1,nx+1)';  % spatial domain 

% right-hand side
app.b = @(x,t) (-sin(pi*x)*(sin(t) - a*pi^2*cos(t)));

% analytic solution
app.u_exact = zeros(nx+1,nt+1);
for j = 1:nx+1
    for i = 1:nt+1
        app.u_exact(j,i) = sin(pi*x(j))*cos(t(i));
    end
end
figure;
surf(x,t,app.u_exact');
title('analytic solution')
xlabel('x'); ylabel('t');
set(gca,options_labels{:})

%% MGRIT parameters
m     = 2;                 % coarsening factor
L     = 2;                 % number of grid levels
maxit = 10;                % maximum number of iterations
tol   = 1e-7;              % stopping tolerance

% time values at each grid level
tc = cell(L,1);
for l=1:L
    tc{l} = t(1:m^(l-1):end);
end

% setup matrix that acts in space for time integrator Phi
app.M = cell(L,1);
e     = ones(nx-1,1);
for l=1:L
    r         = (dt*m^(l-1))/(dx^2);
    app.M{l}  = spdiags([-a*r*e (1+2*a*r)*e -a*r*e], -1:1, nx-1, nx-1);
end

% time integrator
Phi = @(ustart, tstop, tstart, app, l) ...
          (app.M{l}\(ustart + (tstop-tstart)*app.b(x(2:end-1),tstop)));
                                      
% solution and FAS right-hand side on each grid level - without boundary
% dofs!
u = cell(L,1);
v = cell(L,1);
g = cell(L,1);
for l=1:L
    u{l} = zeros(nx-1,numel(tc{l}));
    v{l} = zeros(size(u{l}));
    g{l} = zeros(size(u{l}));
end
% initial condition
for j = 1:nx-1
    u{1}(j,1) = sin(pi*x(j+1));
end

% residual at each iteration
res  = zeros(1,maxit+1);
iter = maxit;

% compute initial space-time residual
% solve A(u) = g with
%      |   I                |
%  A = | -Phi   I           |
%      |       ...   ...    |
%      |            -Phi  I | 
% where Phi propagates u_{i-1} from t = t_{i-1} to t = t_i:
%    u_i = Phi(u_{i-1}) (including forcing from RHS of PDE)
% and with
%    g = (u_0 0 ... 0)^T
%
% The residual can be computed by
%  r_i = Phi(u_{i-1}) - u_i, i = 1, .... nt,
%  r_0 = 0
r = zeros(size(u{1}));
for i=2:length(t)
    r(:,i) = Phi(u{1}(:,i-1),t(i),t(i-1),app,1) - u{1}(:,i);
end
res(1) = norm(r);
fprintf(1,'iter  0: norm(r) = %e\n', res(1));

%% MGRIT iterations
for nu = 1:maxit
    [u,v,g] = mgrit(1,L,u,v,g,Phi,app,m,tc,nu);
    
    % compute space-time residual
    r = zeros(size(u{1}));
    for i=2:length(t)
        r(:,i) = Phi(u{1}(:,i-1),t(i),t(i-1),app,1) - u{1}(:,i);
    end
    res(nu+1) = norm(r);
    fprintf(1,'iter %2d: norm(r) = %e\n', nu, res(nu+1));
    if res(nu+1) < tol
        iter = nu;
        break;
    end
end

%% plot residual reduction
figure;
semilogy(0:iter,res(1:iter+1),'*-',options_plot{:});
title([int2str(L),'-level MGRIT'])
set(gca,options_labels{:})

%% plot solution
% first add (zero) boundary
u_comp = zeros(numel(x),numel(tc{1}));
u_comp(2:nx,:) = u{1};
figure;
surf(x,t,u_comp');
title('computed solution')
xlabel('x'); ylabel('t');
set(gca,options_labels{:})

