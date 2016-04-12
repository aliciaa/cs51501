function [Y, Thi] = tracemin_body(A, B, n, s)
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
% input:  A, B, n, s
% output: Y, Thi
% 
% A, B : n x n sparse mtx 
% n    : matrix dimension, we assume [n,n] == size(A) == size(B)
% s    : no. of eigenvalues we want (block size) 
%                                   (default s = 2 times ni)
% Y    : s x s sparse? mtx          (eigenvectors of H)
% Thi  : s x s diag mtx             (eigenvalues  of H) (ascending order)
% 
%===============================================================


if nargin==0
    disp('DEBUG MODEL');
    A   = sparse(rand(10));
    A   = A+A'+10*sparse(eye(10));
    B   = sparse(rand(10));
    B   = B+B'+10*sparse(eye(10));
    n   = 10;
    s   = 5;
elseif nargin ~= 5
    disp('usage: [X,bStop, Y, Thi] = tracemin_body(A, B, n, s)');
    disp('   or  [X,bStop, Y, Thi] = tracemin_body()');
    return
end

k=1; bStop=0;  thold = 10^(-6);     %threashold 

V = Jacobi_algo_func1(B, n, s);   %FIXME: definitedly needed to be fixed

while 1 
  W=A*V;   %FIXME if necessary
  H=V'*W;  %FIXME if necessary
  [Y,Thi] = Jacobi_algo_func2(H); %FIXME: definitely needed to be fixed
  X=V*Y;
  R=W*Y-B*X*Thi;
  [row,col,v] = find(R);
  for k = 1 : s
    i=find(col==k);
    if v(i)'*v(i) <= Thi(k,k)*thold 
        bStop=0; break;
    end
  end
  if(bStop==1)
    return
  end
  delt = mCG_solver(A,B,X);
  V = Jacobi_algo_B_orth(X-delt);  %FIXME: definitely needed to be fixed
end 

end %end of function

