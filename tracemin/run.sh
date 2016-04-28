#!/bin/tcsh

if ($#argv == 0) then
  ./tracemin -fA data/A_tiny.mtx -fB data/B_tiny.mtx -fO eigen_minres.res
else if ($1 == "d") then
  ./tracemin_deflation -fA data/A_tiny.mtx -fB data/B_tiny.mtx -fO eigen_minres_d.res
endif
