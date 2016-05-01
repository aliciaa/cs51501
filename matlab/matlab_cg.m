
function [solutions] = matlab_cg(A,Q1,RHS, Sig)

  [n,s] = size(RHS);
  A_ = (eye(n)-Q1*Q1')*A;
  for j=1:s
      solutions(:,j) = pcg(A_, RHS(:,j), 1e-3, 100);
  end
end

