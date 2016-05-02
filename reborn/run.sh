#!/bin/tcsh

if ($#argv == 1) then
  ./tracemin -p $1 -fA ../data/A_100.mtx -fB ../data/B_100.mtx -fO eigen_minres
else if ($2 == "d") then
  ./tracemin_deflation -p $1 -fA ../data/A_100.mtx -fB ../data/B_100.mtx -fO eigen_minres_d
else if ($2 == "v") then
  ./tracemin_davidson -p $1 -fA ../data/A_100.mtx -fB ../data/B_100.mtx -fO eigen_minres_v
endif
