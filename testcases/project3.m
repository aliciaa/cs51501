%% cond(A) cond(B)
clc
clear
n=5000;
A=sprandn(n,n,0.001);
B=sprandn(n,n,0.001);
A=A+A';
B=B+B';
alpha=eigs(B,1,'sa');
%
B=B+speye(n)*(abs(alpha)+10);
%lambda=eig(full(B));
%min(lambda)

nnz(A)/n/n
nnz(B)/n/n
%
field='real';
precision=8; 

output='A_4.mtx';
comment = str2mat('matrix A');
[err] = mmwrite(output,A,comment,field,precision);

output='B_4.mtx';
comment = str2mat('matrix B');
[err] = mmwrite(output,B,comment,field,precision);