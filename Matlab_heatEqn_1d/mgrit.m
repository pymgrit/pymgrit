function [u,v,g] = mgrit(l, L, u, v, g, Phi, app, m, t, iter)

if l == L
    %% coarse-grid solve (or L=1): time stepping
    for i=2:length(t{l})
        u{l}(:,i) = g{l}(:,i) + ...
                     feval(Phi,u{l}(:,i-1),t{l}(i),t{l}(i-1),app,l);
    end
else
    % MGRIT algorithm
    %% F- and C-points on current grid level
    nt           = length(t{l});
    Cpts         = 1:m:nt;
    Fpts         = 1:nt;
    Fpts(Cpts)   = [];

    % fprintf(1,'********* start MGRIT *********\n');
    % fprintf(1,'  l                  = %d\n', l);
    % fprintf(1,'  nt                 = %d\n', nt);
    % fprintf(1,'  m                  = %d\n', m);
    % fprintf(1,'  number of F-points = %d\n', length(Fpts));
    % fprintf(1,'  number of C-points = %d\n', length(Cpts));


    %% F-relaxation
    if ((l > 1) || ( (iter == 1) && (l == 1) ))
        for i=Fpts
            u{l}(:,i) = g{l}(:,i) + ...
                         feval(Phi,u{l}(:,i-1),t{l}(i),t{l}(i-1),app,l);
        end
    end

    %% C-relaxation
    for i=Cpts(2:end)
       u{l}(:,i) = g{l}(:,i) + ...
                    feval(Phi,u{l}(:,i-1),t{l}(i),t{l}(i-1),app,l);
    end

    %% F-relaxation
    for i=Fpts
       u{l}(:,i) = g{l}(:,i) + ...
                    feval(Phi,u{l}(:,i-1),t{l}(i),t{l}(i-1),app,l);
    end

    %% inject the fine-grid approximation and compute FAS right-hand side
    v{l+1} = u{l}(:,Cpts);
    % FAS right-hand side: R(g) + A_c( R(u) ) - R( A(u) )
    for i=2:length(Cpts)
        g{l+1}(:,i) = g{l}(:,Cpts(i)) + ...
                       v{l+1}(:,i) - ...
                       feval(Phi,v{l+1}(:,i-1),t{l+1}(i),t{l+1}(i-1), ...
                             app,l+1) - ...
                       u{l}(:,Cpts(i)) + ...
                       feval(Phi,u{l}(:,Cpts(i)-1),t{l}(Cpts(i)), ...
                             t{l}(Cpts(i)-1),app,l);
    end

    %% next level
    u{l+1}  = v{l+1};
    [u,v,g] = mgrit(l+1, L, u, v, g, Phi, app, m, t, iter);
    
    %% correct the approximation u at C-points
    u{l}(:,Cpts) = u{l}(:,Cpts) + (u{l+1}-v{l+1});
    
    %% carry out F-relax to correct the approximation u at F-points
    for i=Fpts
        u{l}(:,i) = g{l}(:,i) + ...
                     feval(Phi,u{l}(:,i-1),t{l}(i),t{l}(i-1),app,l);
    end
end
