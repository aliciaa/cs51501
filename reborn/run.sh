#!/bin/tcsh

if ($#argv == 1) then
  ./tracemin -p $1 -fA ../data/A_tiny.mtx -fB ../data/B_tiny.mtx -fO eigen_minres.res
else if ($2 == "d") then
  ./tracemin_deflation -p $1 -fA data/bcsstk09.mtx -fB data/bcsstk09_B.mtx -fO eigen_minres_d.res
endif
