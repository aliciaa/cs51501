function [Y, Thi] = tracemin_body(A, B, k, ui, uj)
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
% input:  A, B, k, ui, uj
% output: Thi, Y
% 
% A, B : n x n sparse mtx 
% k    : no. of eigenvalues we want (block size) (default s = 2k)
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
    k   = 5;
elseif nargin == 3 
    disp('TraceMin body without region')
elseif nargin == 5
    disp('TraceMin body with region')
    A = A-(ui+uj)/2*B;
else    
    disp('usage: [Thi, Y] = tracemin_body(A, B, k, ui, uj)');
    disp('       [Thi, Y] = tracemin_body(A, B, k)');
    disp('   or  [Thi, Y] = tracemin_body()');
    return
end

s = 2*k;
[n,n] = size(A);
THRESHOLD = 1e-6;           %threshold 
Z = eye(n,s);

while 1
  [Q,Sigma] = eig(Z'*B*Z);
  V = Z*Q/sqrt(Sigma);
  W=A*V;                     
  H=V'*W;                    
  [Y,Thi] = eig(H);
  [S, idx] = sort(diag(Thi));
  Thi = diag(S);
  Y = Y(:,idx);
  X=V*Y;
  R=W*Y-B*X*Thi;
  bStop = 1;
  for col = 1 : k
    if norm(R(:,col),2) > Thi(col,col)*THRESHOLD 
        bStop=0; break;        %if any column does not meet threshold, then continue
    end
  end
  if bStop == 1
    break;  % break the while loop
  end
  Delta = mCG_solver(A,B,X, Thi);    %Thi is used to determin step m within CG. refered from the 1982 PAPER         
  Z = X - Delta;    %FIXME: TODO
end 

X = X(:,1:k);           % only keep the k smallest eigenvectors
Thi = Thi(1:k,1:k);     % only keep the k smallest eigenvalues

end %end of function
