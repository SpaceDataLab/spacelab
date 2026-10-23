! Evaluate NRL HWM14 at points supplied on standard input.
!
! This program is ours; hwm14.f90 is NRL's and is compiled unmodified beside it (see
! tools/fetch-hwm14.sh for why it is fetched rather than vendored). Calling a subroutine is not a
! derivative of it, which is the whole reason the oracle takes this shape instead of a port.
!
! Input, one request per line, whitespace separated:
!
!   iyd  sec  alt_km  glat_deg  glon_deg  stl_hours  f107a  f107  ap3h
!
! Output, one line per request, the two components HWM14 returns:
!
!   meridional_m_s  zonal_m_s
!
! Meridional is positive northward and zonal positive eastward, which is HWM14's own convention and
! the opposite order from the (east, north) this repo's wind model uses -- the swap is done at the
! consumer, once, rather than silently here.
!
! Reads are checked: a malformed line ends the run with a non-zero status rather than being skipped,
! because a comparison that silently drops the rows it could not parse is a comparison on a
! different data set than the one it claims.

program hwm14_oracle
   use, intrinsic :: iso_fortran_env, only: error_unit
   implicit none

   integer :: iyd, ios, line
   real(4) :: sec, alt, glat, glon, stl, f107a, f107, ap(2), w(2)

   line = 0
   do
      read (*, *, iostat=ios) iyd, sec, alt, glat, glon, stl, f107a, f107, ap(2)
      if (ios < 0) exit
      line = line + 1
      if (ios > 0) then
         write (error_unit, '(a,i0)') 'hwm14_oracle: malformed request on line ', line
         stop 1
      end if
      ! HWM14 documents ap(1) as unused and ap(2) as the current 3-hourly ap index.
      ap(1) = -1.0
      call hwm14(iyd, sec, alt, glat, glon, stl, f107a, f107, ap, w)
      write (*, '(2(1x,es24.16))') w(1), w(2)
   end do

end program hwm14_oracle
