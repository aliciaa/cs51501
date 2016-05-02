clear;
n = 108384; b = 1;
output_A='A_108384.mtx';
output_B='B_108384.mtx';
output_C='C_108384.mtx';

A = sparse(n);
B = sparse(n);
C = sparse(n);
for i = 1:n
%    A(i,i) = n * i;
    B(i,i) = 4;
    C(i,i) = 1;
    if (i~=1)
        B(i, i-1) = -1;
    end
    if (i~=n)
        B(i, i+1) = -1;
    end
%    for j = 1:b
%        if (i-j>=1)
%            A(i,i-j) = j;
%        end
%        if (i+j<=n)
%            A(i,i+j) = j;
%        end
%    end
end

field='real';
precision=8;


comment = str2mat('matrix A');
[err] = mmwrite(output_A,A,comment,field,precision);


comment = str2mat('matrix B');
[err] = mmwrite(output_B,B,comment,field,precision);

comment = str2mat('matrix C');
[err] = mmwrite(output_C,C,comment,field,precision);
