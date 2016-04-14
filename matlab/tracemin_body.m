function [X, Thi] = tracemin_body(A, B, s, ui, uj)
%% basic trace minimization alg.
% Alg 11.13 of book "Parallelism in Matrix Computations"
% 
% April 12th 2016 Project 3 of CS51501
%
% L1: Choose a block size s>=p and an n*s matrix V of full rank such that V'BV = I
% L2: do k=1,2,...until convergence
% L3:     Compute W=AK and the interaction matrix H=V'W
% L4:     Compute the eigenpaires (Y, Thi) of H. (Thi should be arranged asceding order, Y be orthogonal)
% L5:     Compute the corresponding Ritz vector X = V*Y   
% L6:     Compute the residuals R = WY - BXThi =(AX-BXThi)
% L7:     Test for convergence
% L8:     Solve the positive-semidefinite linear system(11.66) approximately via the CG scheme
% L9:     B-orthonormalize X-=delt 
%============================================================
%
% input:  A, B, n, s, ui, uj
% output: Thi, Y
% 
% A, B : n x n sparse mtx 
% n    : matrix dimension, we assume [n,n] == size(A) == size(B)
% s    : no. of eigenvalues we want (block size) (default s = 2ni)
% ui,uj: region [ui,uj]             (used to form new A-=0.5*(ui+uj)*B)
% Thi  : s x s diag mtx             (eigenvalues  of H) (ascending order)
% Y    : s x s sparse? mtx          (eigenvectors of H)
% 
%===============================================================


if nargin==0
    disp('tracemin_body DEBUG MODEL');
    A   = sparse(rand(10));
    A   = A+A'+10*sparse(eye(10));
    B   = sparse(rand(10));
    B   = B+B'+10*sparse(eye(10));
    s   = 5;
elseif nargin == 4
    disp('TraceMin body without region')
elseif nargin == 6
    disp('TraceMin body with region')
    A = A-(ui+uj)/2*B;
else    
    disp('usage: [Thi, Y] = tracemin_body(A, B, s, ui, uj)');
    disp('       [Thi, Y] = tracemin_body(A, B, s)');
    disp('   or  [Thi, Y] = tracemin_body()');
    return
end

[n,n] = size(A);
THRESHOLD = 10^(-6);           %threshold 
Z = eye(n, s);

while 1
  [Q,Sigma] = eig(B*Z);
  V = Z*Q/sqrt(Sigma);
  W=A*V;                     
  H=V'*W;                    
  [Y,Thi] = eig(H); 
  X=V*Y;
  R=W*Y-B*X*Thi;
  bStop = 1;
  for k = 1 : s
    if norm(R(:,k),2) > Thi(k,k)*THRESHOLD 
        bStop=0; break;        %if any column does not meet threshold, then continue
    end
  end
  if bStop == 1
    break;  % break the while loop
  end
  delt = mCG_solver(A,B,X,n,s,k);
  Z = X - delt;    %FIXME: TODO
end 

end %end of function
