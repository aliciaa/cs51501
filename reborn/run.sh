#!/bin/tcsh

if ($#argv == 1) then
  ./tracemin -p $1 -fA ../data/A_100.mtx -fB ../data/B_100.mtx -fO eigen_minres.res
else if ($2 == "d") then
  ./tracemin_deflation -p $1 -fA ../data/A_100.mtx -fB ../data/B_100.mtx -fO eigen_minres_d.res
endif
