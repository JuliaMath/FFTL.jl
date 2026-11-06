! The legacy routines under g77's name mangling (two trailing underscores), called through
! explicit bind(C) names; arguments by reference, as Fortran passes them.
subroutine run_legacy_mangling()
  use, intrinsic :: iso_c_binding
  use ftest_ref
  implicit none
  interface
    subroutine plan_dft_1d_g77(plan, n, in, out, sign, flags) bind(C, name='dfftw_plan_dft_1d__')
      import
      integer(C_INT64_T) :: plan
      integer(C_INT) :: n, sign, flags
      complex(C_DOUBLE_COMPLEX) :: in(*), out(*)
    end subroutine plan_dft_1d_g77
    subroutine execute_dft_g77(plan, in, out) bind(C, name='dfftw_execute_dft__')
      import
      integer(C_INT64_T) :: plan
      complex(C_DOUBLE_COMPLEX) :: in(*), out(*)
    end subroutine execute_dft_g77
    subroutine destroy_plan_g77(plan) bind(C, name='dfftw_destroy_plan__')
      import
      integer(C_INT64_T) :: plan
    end subroutine destroy_plan_g77
  end interface
  integer(C_INT64_T) :: plan
  complex(C_DOUBLE_COMPLEX) :: x(9), y(9)
  complex(8) :: ref(9)
  integer :: i
  do i = 1, 9
    x(i) = cmplx(i, -2 * i, C_DOUBLE_COMPLEX)
  end do
  ref = x
  call dft_dim(ref, [9], 1, 1, -1)
  call plan_dft_1d_g77(plan, 9, x, y, -1, 64)
  call execute_dft_g77(plan, x, y)
  call check(relerr_c(y, ref, 9) < 1d-12, 'g77 mangling: dfftw_plan_dft_1d__, dfftw_execute_dft__')
  call destroy_plan_g77(plan)
end subroutine run_legacy_mangling
