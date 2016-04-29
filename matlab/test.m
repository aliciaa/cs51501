

function test()
load('nos4.mat');
A = full(Problem.A);
[d, d] = size(A);
B = 4*diag(ones(d,1)) + (-1)*diag(ones(d-1,1),1) + (-1)*diag(ones(d-1,1),-1);
lower_bound = 0;
upper_bound = 1;
num_of_intervals=2;
[num_of_eig, intervals] = multi_section(A, B, lower_bound, upper_bound, num_of_intervals);
tracemin_multi_section(A, B, num_of_eig(1), intervals(1), intervals(2));

