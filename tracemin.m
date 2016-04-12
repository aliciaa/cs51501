function [Y, Thi] = tracemin(A, B, s, p)
%% basic trace minimization alg. 
% Alg 11.13 of book "Parallelism in Matrix Computations"
% 
% April 12nd 2016 Project 3 of CS51501
% 
%============================================================
%
% input:  A, B, p
% output: Y, Thi
% 
% A, B : n x n sparse mtx 
% s    : no. of eigenpairs we want (smallest s eigenpairs), s<<n
% p    : no. of processes. or no. of sections
% Y    : n x s sparse mtx                (eigenvectors of A)
% Thi  : s x s sparse diag. mtx (diag.)  (eigenvalues of  A)
%===============================================================
%
%

if nargin==0
  disp('DEBUG MODEL')
  A = sparse(rand(10));
  A = A+A'+10*sparse(eye(10));
  B = sparse(rand(10));
  B = B+B'+10*sparse(eye(10));
  s = 8;
  p = 2;
elif nargin != 4
  disp('usage: [Y, Thi] = tracemin(A, B, s, p');
  disp('   or  [Y, Thi] = tracemin()');
  return
end



%%TODO1 multi section 
% .. ..
[p,A ...]=TODO_fun1();

%%trace mini body
X = tracemin_body(A, B, 2*p) 

return 
