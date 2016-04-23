function [delt] = mCG_solver(A, B, X, Thi)
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


if nargin==0
    disp('mCG_solver DEBUG MODEL');
    n = 10;
    s = 2;
    k = 1;
    A = sparse(rand(n,n)); A=A+A'+10*sparse(eye(n));
    B = sparse(rand(n,n)); A=A+A'+10*sparse(eye(n));
    X = sparse(rand(n,s));
elseif nargin == 4
    disp('mCG_solver()');
else
    disp('usage: [Y, Thi] = tracemin(A, B, s, p');
    disp('   or  [Y, Thi] = tracemin(A, B, s, p, a, b');
    disp('   or  [Y, Thi] = tracemin()');
    return
end

[n,s] = size(X)
POLICY = 3;
tol = somehowget_tolerance(k);  % CG should be rough at the beginning and accurate in the end


if POLICY == 1
    P = getP(B,X); 
    A_ = P*A*P;
    D_ = P*A*X;
    delt = sparse(n,s);
    for i = 1:s
        delt(:,i) = pcg(A_, D_(:,i),  tol);
    end
    return
    
elseif POLICY ==2
    P = getP(B,X);
    delt = pcg(@afun, reshape(P*(A*X),[n*s,1]), tol);  % a function handle, afun, such that afun(x) returns P*A*x 
    delt = sparse(reshape(delt, [n,s]));
    return

else %POLICY >=3
    P = getP(B,X);
    delt = mCG_core(P*A, P*A*X, n, s, Thi);   %Alg 3. CG s.t. accept matrix
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
