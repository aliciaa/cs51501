####Our TraceMin based on PETSc and ??

####How to use
* Install PETSc, configuer and make
* Download *cg_seq.c* move and rename it to *cg.c* into  
  <code>  &lt;PETSc_Folder&gt;/scr/ksp/ksp/impls/cg/cg.c </code>  
  go to <code>&lt;PETSc_Folder&gt;/</code> and <code>make</code> 
* Download all the other files into  
  <code>  &lt;PETSc_Folder&gt;/scr/ksp/ksp/examples/tutorials/ </code>  
  And<pre>
$make tracemin
./run.sh
</pre>



#### Where is QR factorilization
* <code>int getQ1(A,M,N); </code> in *tracemin_qr.cpp*.   
where A is matrix to be QR factorilization  
A would be overwrited with the result Q1  
A is a M by N matrix;


