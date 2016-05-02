#!/bin/tcsh

#if ($#argv == 0) then
  ./tracemin -fA A.mtx -fB B.mtx -fO eigen_minres.res
  #else if ($1 == "d") then
    #  ./tracemin_deflation -fA data/A.mtx -fB data/B.mtx -fO eigen_minres_d.res
    #endif
