! The Fortran test of the FFTW3 interface: the legacy (dfftw_*, sfftw_*) and Fortran 2003
! (fftw_*, fftwf_* through fftw3.f03) interfaces in both precisions against direct sums.
! Built by fftw3_ftest.sh against FFTW and against libfftl.
program ftest
  use ftest_ref
  implicit none
  call run_legacy_d()
  call run_legacy_s()
  call run_legacy_mangling()
  call run_f03_d()
  call run_f03_s()
  print '(a,i0,a,i0,a)', 'fftw3_ftest: ', ntests, ' checks, ', nfailed, ' failed'
  if (nfailed /= 0) stop 1
end program ftest
