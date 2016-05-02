%


fA = '../testcases/A30.mtx';
fB = '../testcases/B30.mtx';
p=2;






A = mmread(fA);
B = mmread(fB);
[EigVec, EigVal] = tracemin(A,B,p);

