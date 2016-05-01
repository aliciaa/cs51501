%tracemin basic
function [Y, Thi] = tracemin(A, B, k)

if nargin~=3
    disp('       [Thi, Y] = tracemin_body(A, B, k)');
    return
end
disp('TraceMin body without region')

s = 2*k;
[n,~] = size(A);
THRESHOLD = 1e-6;           %threshold 
Z = zeros(n, s);
for i = 1:n
    Z(i,mod(i-1,s)+1) = 1.0;
end
cnt =1;
while 1
  [Q,Sigma] = eig(Z'*B*Z);
  V = Z*Q/sqrt(Sigma);
  W=A*V;                     
  H=V'*W;                    
  [EigY,Thi] = eig(H);
  [S, idx] = sort(diag(Thi));
  Thi = diag(S);
  EigY = EigY(:,idx);
  
  Y=V*EigY;
  R=W*EigY-B*Y*Thi;
  bStop = 1;
  for col = 1 : k
    if norm(R(:,col),2) > Thi(col,col)*THRESHOLD 
        bStop=0; break;        %if any column does not meet threshold, then continue
    end
  end
  if bStop == 1
    break;  % break the while loop
  end

  %%=== QR factorilization ====
  BY = B*Y;
  [Q,R] = qr(BY);
  Q1=Q(:,1:s);
  RHS = i_qqax(A, Q1, Y);   % (I-QQ')AY
  
  %%=== Linear Solver      =====
  Solutions = linear_solver_cg(A,Q1,RHS,Thi);
  %Solutions = matlab_cg(A,Q1,RHS,Thi);
  
  Z = Y - Solutions; 
  cnt=cnt+1;
end 

Y = Y(:,1:k);           % only keep the k smallest eigenvectors
Thi = Thi(1:k,1:k);     % only keep the k smallest eigenvalues



    % function performance  (I-QQ')AY
    function RHS = i_qqax(A, Q1, Y)
      AY = A*Y;
      RHS = AY - Q1*(Q1'*AY);
    end



end %end of function
