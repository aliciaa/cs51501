function [num_of_eigs, intervals] = multi_section(A, B, lower_bound, upper_bound, num_of_intervals)
num_of_eigs = zeros(num_of_intervals, 1);
intervals= zeros(num_of_intervals+1, 1);
intervals(1) = lower_bound;
int_min_length = 0.001;
[lower_neg, lower_pos] = countd(A, B, lower_bound);
[upper_neg, upper_pos] = countd(A, B, upper_bound);
neg = lower_neg;
total = upper_neg - lower_neg;
average = total / num_of_intervals;
for i = 1:num_of_intervals - 1
    int_lower_bound = intervals(i);
    int_upper_bound = upper_bound;
    prev_neg = neg;
    flag = 0;
    while (flag == 0)
        int_mid_bound = (int_lower_bound + int_upper_bound) / 2;
        [neg, pos] = countd(A, B, int_mid_bound);
        %neg - prev_neg;
        %if abs(neg - prev_neg - average) <= 2
        if abs(neg-prev_neg-average)/average <=0.05
            flag = 1;
            num_of_eigs(i) = neg - prev_neg;
            intervals(i+1) = (int_mid_bound);
            total = total - (neg - prev_neg);
            average = total / (num_of_intervals - i);
        elseif (int_mid_bound - int_lower_bound < int_min_length)
            flag = 1;
            num_of_eigs(i) = neg - prev_neg;
            intervals(i+1) = (int_mid_bound);
            total = total - (neg - prev_neg);
            average = total / (num_of_intervals - i);
        elseif neg - prev_neg < average
            int_lower_bound = int_mid_bound;
        else
            int_upper_bound = int_mid_bound;
        end
    end
end
num_of_eigs(num_of_intervals) = upper_neg - neg;
intervals(num_of_intervals + 1) = upper_bound;