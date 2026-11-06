! Reference transforms (direct sums in double precision) and the check counter of the Fortran
! tests of the FFTW3 interface (fftw3_ftest.sh).
module ftest_ref
  implicit none
  integer :: ntests = 0, nfailed = 0
  real(8), parameter :: pi = 3.14159265358979323846264338327950288d0
contains
  subroutine check(cond, name)
    logical, intent(in) :: cond
    character(*), intent(in) :: name
    ntests = ntests + 1
    if (.not. cond) then
      nfailed = nfailed + 1
      print '(a,a)', 'FAIL ', name
    end if
  end subroutine check

  ! DFT along dimension d (1-based) of the column-major array a of size dims(1:k), in place
  subroutine dft_dim(a, dims, k, d, sgn)
    integer, intent(in) :: k, dims(k), d, sgn
    complex(8), intent(inout) :: a(*)
    integer :: s, n, outer, o, l, j, m, base
    complex(8), allocatable :: line(:), res(:)
    s = product(dims(1:d-1))
    n = dims(d)
    outer = product(dims(d+1:k))
    allocate(line(n), res(n))
    do o = 0, outer - 1
      do l = 0, s - 1
        base = o * s * n + l
        do j = 0, n - 1
          line(j + 1) = a(base + j * s + 1)
        end do
        do m = 0, n - 1
          res(m + 1) = (0d0, 0d0)
          do j = 0, n - 1
            res(m + 1) = res(m + 1) + line(j + 1) * exp(cmplx(0d0, sgn * 2d0 * pi * mod(j * m, n) / n, 8))
          end do
        end do
        do j = 0, n - 1
          a(base + j * s + 1) = res(j + 1)
        end do
      end do
    end do
  end subroutine dft_dim

  ! One real-to-real line of FFTW's kind (R2HC = 0 ... RODFT11 = 10)
  subroutine r2r_line(x, y, n, kind)
    integer, intent(in) :: n, kind
    real(8), intent(in) :: x(0:n-1)
    real(8), intent(out) :: y(0:n-1)
    integer :: j, k
    real(8) :: s
    do k = 0, n - 1
      s = 0
      select case (kind)
      case (0)
        if (2 * k <= n) then
          do j = 0, n - 1
            s = s + x(j) * cos(2 * pi * mod(j * k, n) / n)
          end do
        else
          do j = 0, n - 1
            s = s - x(j) * sin(2 * pi * mod(j * (n - k), n) / n)
          end do
        end if
      case (1)
        s = x(0)
        do j = 1, (n - 1) / 2
          s = s + 2 * (x(j) * cos(2 * pi * mod(j * k, n) / n) - x(n - j) * sin(2 * pi * mod(j * k, n) / n))
        end do
        if (mod(n, 2) == 0) s = s + x(n / 2) * (-1)**k
      case (2)
        do j = 0, n - 1
          s = s + x(j) * (cos(2 * pi * mod(j * k, n) / n) + sin(2 * pi * mod(j * k, n) / n))
        end do
      case (3)
        s = x(0) + (-1)**k * x(n - 1)
        do j = 1, n - 2
          s = s + 2 * x(j) * cos(pi * j * k / (n - 1))
        end do
      case (4)
        s = x(0)
        do j = 1, n - 1
          s = s + 2 * x(j) * cos(pi * j * (k + 0.5d0) / n)
        end do
      case (5)
        do j = 0, n - 1
          s = s + 2 * x(j) * cos(pi * (j + 0.5d0) * k / n)
        end do
      case (6)
        do j = 0, n - 1
          s = s + 2 * x(j) * cos(pi * (j + 0.5d0) * (k + 0.5d0) / n)
        end do
      case (7)
        do j = 0, n - 1
          s = s + 2 * x(j) * sin(pi * (j + 1) * (k + 1) / (n + 1d0))
        end do
      case (8)
        s = (-1)**k * x(n - 1)
        do j = 0, n - 2
          s = s + 2 * x(j) * sin(pi * (j + 1) * (k + 0.5d0) / n)
        end do
      case (9)
        do j = 0, n - 1
          s = s + 2 * x(j) * sin(pi * (j + 0.5d0) * (k + 1) / n)
        end do
      case (10)
        do j = 0, n - 1
          s = s + 2 * x(j) * sin(pi * (j + 0.5d0) * (k + 0.5d0) / n)
        end do
      end select
      y(k) = s
    end do
  end subroutine r2r_line

  ! r2r of kind along dimension d of the column-major real array a of size dims(1:k)
  subroutine r2r_dim(a, dims, k, d, kind)
    integer, intent(in) :: k, dims(k), d, kind
    real(8), intent(inout) :: a(*)
    integer :: s, n, outer, o, l, j, base
    real(8), allocatable :: x(:), y(:)
    s = product(dims(1:d-1))
    n = dims(d)
    outer = product(dims(d+1:k))
    allocate(x(n), y(n))
    do o = 0, outer - 1
      do l = 0, s - 1
        base = o * s * n + l
        do j = 0, n - 1
          x(j + 1) = a(base + j * s + 1)
        end do
        call r2r_line(x, y, n, kind)
        do j = 0, n - 1
          a(base + j * s + 1) = y(j + 1)
        end do
      end do
    end do
  end subroutine r2r_dim

  real(8) function relerr_c(y, ref, n)
    integer, intent(in) :: n
    complex(8), intent(in) :: y(n), ref(n)
    relerr_c = maxval(abs(y - ref)) / max(maxval(abs(ref)), tiny(1d0))
  end function relerr_c

  real(8) function relerr_r(y, ref, n)
    integer, intent(in) :: n
    real(8), intent(in) :: y(n), ref(n)
    relerr_r = maxval(abs(y - ref)) / max(maxval(abs(ref)), tiny(1d0))
  end function relerr_r

  ! The first m rows of the n1 x n2 column-major array a, as a vector
  function head_rows(a, n1, n2, m) result(b)
    integer, intent(in) :: n1, n2, m
    complex(8), intent(in) :: a(n1, n2)
    complex(8) :: b(m * n2)
    b = reshape(a(1:m, :), [m * n2])
  end function head_rows

  ! deterministic values in [-1, 1)
  subroutine fill_r(a, n, seed)
    integer, intent(in) :: n
    integer, intent(inout) :: seed
    real(8), intent(out) :: a(n)
    integer :: i
    do i = 1, n
      seed = int(mod(int(seed, 8) * 1103515245_8 + 12345_8, 2147483647_8))
      a(i) = 2d0 * seed / 2147483647d0 - 1d0
    end do
  end subroutine fill_r
end module ftest_ref
