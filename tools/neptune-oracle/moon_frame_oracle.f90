! Print libslam's Moon body-fixed rotation matrix at epochs read from stdin.
!
! NEPTUNE evaluates the lunar harmonics in whatever frame libslam's Reduction_moon_type produces,
! and that choice is not a detail: the Moon's principal-axis frame and its mean-Earth frame differ
! by about 875 m on the surface, so a comparison that assumes one and gets the other reports a
! model error that is really a frame error. This driver asks libslam directly rather than
! reimplementing its series, because a reimplementation would measure the transcription.
!
! Input:  one MJD (TDB) per line.
! Output: one line per epoch -- mjd followed by the nine elements of R in row-major order, where
!         r_body_fixed = R * r_mcrf.

program moon_frame_oracle

  use slam_moon_reduction, only: Reduction_moon_type
  use slam_types,          only: dp

  implicit none

  type(Reduction_moon_type) :: moon_red
  real(dp)                  :: mjd
  real(dp), dimension(3)    :: e1, e2, e3, c1, c2, c3
  integer                   :: status

  ! The rotation matrix is private to the type, so it is recovered a column at a time by rotating
  ! the three basis vectors: column j of R is R applied to the j-th unit vector.
  e1 = [1.d0, 0.d0, 0.d0]
  e2 = [0.d0, 1.d0, 0.d0]
  e3 = [0.d0, 0.d0, 1.d0]

  do
    read (*, *, iostat=status) mjd
    if (status /= 0) exit

    call moon_red%gcrf2moonFixed_r(e1, [0.d0, 0.d0, 0.d0], mjd, c1)
    call moon_red%gcrf2moonFixed_r(e2, [0.d0, 0.d0, 0.d0], mjd, c2)
    call moon_red%gcrf2moonFixed_r(e3, [0.d0, 0.d0, 0.d0], mjd, c3)

    write (*, '(F20.9, 9(1X, ES23.16))') mjd, &
      c1(1), c2(1), c3(1), &
      c1(2), c2(2), c3(2), &
      c1(3), c2(3), c3(3)
  end do

end program moon_frame_oracle
