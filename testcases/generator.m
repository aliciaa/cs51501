clear;
n = 10; b = 2;
A = sparse(n);
B = sparse(n);
for i = 1:n
    A(i,i) = 100;
    B(i,i) = 4;
    if (i~=1)
        B(i, i-1) = -1;
    end
    if (i~=n)
        B(i, i+1) = -1;
    end
    for j = 1:b
        if (i-j>=1)
            A(i,i-j) = j;
        end
        if (i+j<=n)
            A(i,i+j) = j;
        end
    end
end

field='real';
precision=8;

output='A_tiny.mtx';
comment = str2mat('matrix A');
[err] = mmwrite(output,A,comment,field,precision);

output='B_tiny.mtx';
comment = str2mat('matrix B');
[err] = mmwrite(output,B,comment,field,precision);
