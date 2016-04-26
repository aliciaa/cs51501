function [delt] = mCG_solver(A, B, Y, Thi, cnt)
%Solve the positive-semidefinite linear system(11.66) approximately
% via the CG scheme
%
% P A P Delt = P A X      (linear system 11.66)
% s.t.
%     P = I-BX(X'BBX)^(-1)X'B
%     B is S.P.D
%     A is S.P.D
%
%====================================================================
%
% input : A, B, X, n, s, k
% output: delt
% 
% A, B : n x n sparse mtx 
% X    : n x s sparse mtx and s<<n
% n    : no. of rows of X
% s    : no. of columns of X
% k    : out loop iteration cnt
% delt : n x s sparse mtx
%
%====================================================================
% BACK GROUND of this Qusetion
% traceMin want use to solve one of the Quesion
%    %% Question 1
%    %  solve (A, X'B; BX, O)(delt, Lg)=(AX,0)
%    %% Question 2
%    %  solve (A, X'B; BX, O)(X-delt, Lg)=(O, I)
%    %========================================================
%
% If we pick Question 1, it would be equivelent to sove (linear system 11.66)
% since it is just a optimization problem.
% ---------------------------------------------------------------------
% QUESTION!: What if we are trying to pick problem2 ?
%
% =======================================================================
% SCHEME:
% There are three potential problems
% 1. solve P need to inverse
% 2. P Delt = P
% 3. delt is not a vector
% 
% HERE IS AN ALG WITHOUT COSIDERING SUCH Three PROBLEMS
%   %% navie method
%   % P = I-BX(X'BBX)^(-1)X'B
%   % C = PAP
%   % D = PAX
%   % for i in 1:s
%   %    delt(i) = cg(C,D(i))  % C delt = D
%   % end
%   % ------------------------------------------
%   % WASTE LOTS OF TIME
%
% -----------------------------------------------------------------------
% For problem 1. (inverse of P)
%
%   %% Alg 1 to get P without inverse
%   % Q2 = QR_factor(BX)
%   % P  = Q2*Q2'
%   %--------------------------------------------
%   THE REASON
%   %%Let's define C = BX
%   % P = I-C(C'C)^(-1)C'
%   % we know   C = QR  (by QR-factorilizaiton)
%   %      s.t. Q = [Q1,Q2] and R = [R, O]'
%   % since    C'C= ... = R'R
%   % then      P = I-QR(R'R)^(-1)R'Q'
%   %             = I-Q1Q1'
%   %             = Q2Q2'
%   %
%--------------------------------------------------------------------------
% For problem 2. (P delt=P)
%   %% Alg 2 
%   We can only modified CG's inter-face funtion. to do CG by our own
%
%--------------------------------------------------------------------------
% For problem 3. (delt is not a vector) A Delt = B
%   %% Alg 3 to solve delt is not a vector
%   % Reform x = [Delt(:,1)', Delt(:,2)', ..., Delt(:,s)']'
%   %        b = [B(:,1)'   , B(:,2)    , ..., B(:,s)'   ]'
%   %        A = Diag(A     , A         , ..., A         )
%   % sovle Ax=b with CG
% 
% For problem 3. 
%   %% Alg 4 to solve delt is not a vector
%   % modify the CG function. to accept mtx.
%   % need to handle 
%   %               scale = X'RX/X'X   becomes a matrix problme
%
% currently, we implemented: naive alg                   POLICY=1
%                            Alg 1 + Alg 2 + Alg 3       POLICY=2 
%                            Alg 1 + Alg 2 + Alg 4       POLICY=3 (default)
%=====================================================================


if nargin~=5
    disp('mCG_solver(A, B, Y, Thi, cnt)')
    return
end

[n,s] = size(Y);
POLICY = 1;
tol = somehowget_tolerance(cnt);  % CG should be rough at the beginning and accurate in the end


if POLICY == 1
    P = getP(B,Y); 
    A_ = P*A*P;
    D_ = P*A*Y;
    delt = sparse(n,s);
   
    BY=B*Y;
    [Q,R] = qr(BY);
    Q1=Q(:, 1:s);
    mmwrite(sprintf('Q1_step%d.mtx',cnt), Q1);
    mmwrite(sprintf('I_Q1Q1t_step%d.mtx',cnt), eye(n)-Q1*Q1');


    mmwrite(sprintf('P_step%d.mtx',cnt), P);  
    mmwrite(sprintf('PA_step%d.mtx',cnt), P*A);
    mmwrite(sprintf('PAY_step%d.mtx',cnt), P*A*Y);
    
    for i = 1:s
        delt(:,i) = pcg(A_, D_(:,i),  tol);
    end
    return
    
elseif POLICY ==2
    P = getP(B,Y);
    delt = pcg(@afun, reshape(P*(A*Y),[n*s,1]), tol);  % a function handle, afun, such that afun(x) returns P*A*x 
    delt = sparse(reshape(delt, [n,s]));
    return

else %POLICY >=3
    P = getP(B,Y);  
    fname = sprintf('P_step%d.mtx',cnt);
    mmwrite(fname ,P);  
    fname = sprintf('PA_step%d.mtx',cnt);
    mmwrite(fname, P*A);
    fname = sprintf('PAY_step%d.mtx',cnt);
    mmwrite(fname, P*A*Y);
    delt = mCG_core(P*A, P*A*Y, n, s, Thi);   %Alg 3. CG s.t. accept matrix
    return
    
end %end of if
    

    % CG should be rough at the beginning and accurate in the end    
    function tol = somehowget_tolerance(~)
        tol = 10^(-3);
    end

    function P = getP(B,X)
        P = sparse(eye(n)) - B*X* inv(X'*B*B*X) * X' *B
        %[Q,R] = qr(B*X);  % TODO Q may be have too many fill in 
        %Q2 = Q(:,s+1:n);  % Thus We'd better replace this part with our own method
        %P = Q2*Q2';
    end

    function y = afun(x)
        y=zeros(n*s,1);
        for i =1:s
            y((i-1)*n+1: i*n) = P*(A* x((i-1)*n+1:i*n) );
        end
    end
end %end of funtion
