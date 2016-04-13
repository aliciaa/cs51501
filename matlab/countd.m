function [neg, pos] = countd(A, B, shift_value)
Ai = A - shift_value*B;
[L, D] = ldl(full(Ai));
neg = 0;
pos = 0;
for i=1:size(Ai)
    if D(i, i) <= 0
        neg = neg+1;
    else
        pos = pos+1;
    end
end
%disp(sprintf('count neg=', neg));
%disp(sprintf('count pos=', pos));

        