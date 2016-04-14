function [Thi, Y] = tracemin(A, B, s, p, a, b)
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
% a,b  : regions [a,b]
% Thi  : s x s sparse diag. mtx (diag.)  (eigenvalues of  A)
% Y    : n x s sparse mtx                (eigenvectors of A)
%===============================================================
%
%

if nargin==0
  disp('tracemin DEBUG MODEL')
  A = sparse(rand(10));
  A = A+A'+10*sparse(eye(10));
  B = sparse(rand(10));
  B = B+B'+10*sparse(eye(10));
  s = 8;
  p = 2;
  a = 0;
  b = 100;
  bMultiSection = 1;
elseif nargin == 4
  disp('The first problem. (without regions [a,b])');
  bMultiSection = 0
elseif nargin == 6
  disp('The second problem. (with regions [a,b])');
  bMultiSection = 1
else
  disp('usage: [Y, Thi] = tracemin(A, B, s, p');
  disp('   or  [Y, Thi] = tracemin(A, B, s, p, a, b');
  disp('   or  [Y, Thi] = tracemin()');
  return
end



n   = size(A,1);
Y   = sparse(n,n); %empty sparse  matrix waiting to be filled in
Thi = sparse(n,n); %empty diagnal matrix waiting to be filled in 


if bMultiSection == 1
  [ni_list,intervals]=multi_section(A, B, a, b, s);
  ibeg=1; 
  for i = 1 : p
    iend=ibeg+ni_list(1); 
    [Thi(ibeg:iend, ibeg:iend),Y(:,ibeg:iend)] = tracemin_body(A, B, 2*ni_list(i), n, intervals(i), intervals(i+1));
    ibeg=iend+1;
  end
else
  [Thi, Y] = tracemin_body(A, B, 2*s, n); 
end

return 
