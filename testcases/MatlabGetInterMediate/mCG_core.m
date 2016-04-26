function [ x ] = mCG_core(A, b, n, s, Thi)
%% Advanced CG alg
% advanced CG to support Ax=b 
%         with x,b be matrixs
%  
%============================================================
%
% input: A, B, n, s, tol
% output: X
% 
% A    : n x n sparse mtx 
% B    : n x s sparse mtx 
% n    : 1 x 1 scalar, system size
% s    : 1 x 1 scalar no. of linear equations
% tol  : 1 x 1 scalar stop critirial
% X    : n x s sparse mtx
% 
%===============================================================
% Orig CG
% r0 = b-Ax0
% P  = r
% k  = 0
% repeat
%    a = r0'r0/(P'AP)
%    x1= x0+a*P
%    r1= r1-a*A*P
%    if ||r1|| < threashold
%        return
%    b = (r1'r1) / (r0'r0)
%    P = r1+b*P
%===============================================================

if nargin ==0
    disp('mCG_core DEBUG MODEL');
    n = 10;
    s = 2;
    A = sparse(rand(n,n)); A=A+A'+10*sparse(eye(n));
    realX = sparse(rand(n,s));
    b = sparse(A*realX);
    tol = 10^(-3);
    A
    b
    realX

end

 tol = 10^(-3);

%initial guess is zero
x     = sparse(n,s); %sparse
r     = b; %-A*X       
P     = r;
rsold = IPAAcc(r,s);

for i = 1: length(b)
  AP    = A*P;
  alpha = rsold./IPABcc(P,AP,n,s);
  x     = x  + Zoomkv(alpha,P,s);
  r     = r  - Zoomkv(alpha,(A*P),s);
  rsnew = IPAAcc(r,s);
  if sqrt(max(rsnew)) < tol  %TODO, fixme, since m should be estimated according to 1982 papre
      break
  end
  P     = r + Zoomkv((rsnew./rsold),P,s);
  rsold = rsnew;
end

    %Inner Product A x A colum by colum
    function v = IPAAcc(A,s)
        v = sparse(s,1); % zeros(s,1);
        for i = 1:s
            v(i)= A(:,i)'*A(:,i);
        end
    end

    %Inner Product A x B colum by colum
    function v = IPABcc(A,B,~,s)
        v = sparse(s,1); % zeros(s,1);
        for i = 1:s
            v(i) = A(:,i)'*B(:,i);
        end
    end

    %Zoom vector by scalar (k is scalars, P is vectors)
    function P = Zoomkv(k,P,s)
        for i = 1:s
            P(:,i)=k(i)*P(:,i);
        end
    end

end %end of the function
