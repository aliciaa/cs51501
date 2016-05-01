
function [solutions] = linear_solver_cg(A,Q1,RHS, Sig)
  CG_MAX_ITER = 100;
  
  [n,s] = size(RHS);

  solutions = zeros(n,s);
  for j = 1:s
    r = RHS(:,j);
    p = RHS(:,j);
    x = zeros(n,1);
    alpha = 0;
    beta  = 0;
    rnorm = 0;
    rnorm2= 0;
    

    rnorm = norm(r);
    for i=1:CG_MAX_ITER
      Ap = i_qqax(A, Q1, p);
      alpha = (rnorm*rnorm)/ (p'*Ap);
      x = x+alpha*p;
      r = r-alpha*Ap;
      rnorm2 = norm(r);
      if( rnorm2*rnorm2 < 1e-3 )
        break;
      end

      beta = rnorm2*rnorm2/ (rnorm*rnorm);
      p = r + beta*p;
      rnorm = rnorm2;
    end
    solutions(:,j)=x;
  end

    %subfunction of (I-QQ)Ax
    function Ap = i_qqax(A, Q1, x)
      Ax = A*x;
      Ap = Ax-Q1*(Q1'*Ax);
    end

end

