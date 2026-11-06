! Counts the characters of exported wisdom (legacy dfftw_export_wisdom/sfftw_export_wisdom).
subroutine wcount_char(c, iunit)
  implicit none
  character :: c
  integer :: iunit
  integer :: wcount
  common /wisdom_count/ wcount
  wcount = wcount + 1
end subroutine wcount_char
