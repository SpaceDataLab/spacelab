      PROGRAM ORACLE
*  Reference values for the spacelab-slr troposphere and station-displacement ports.
*
*  Every case here is one the IERS routines document in their own headers, so the numbers
*  this prints can be read against both the published values and the Rust port. Where the
*  two disagree -- FCULZD_HPA does, by two parts in 10^6 -- the disagreement is the finding;
*  see README.md.
      IMPLICIT NONE
      DOUBLE PRECISION ZTD, ZHD, ZWD, FCUL_A, M
      DOUBLE PRECISION XSTA(3), XSUN(3), XMON(3), DXTIDE(3)
      INTEGER ITM(5), I, NK
      COMMON/DATE/ITM
      REAL XK(20), UK(20), SK(20), AK(20), Y, EVAL
      EXTERNAL EVAL
      DOUBLE PRECISION XKD(6), UKD(6)
      DATA XKD/0.31D0, 0.62D0, 1.07D0, 1.55D0, 2.03D0, 2.40D0/
      DATA UKD/0.70D0, -1.30D0, 0.45D0, -0.11D0, 0.93D0, -0.28D0/

*  FCULZD_HPA, its own header case: Mendes--Pavlis zenith delay at Monument Peak.
*  latitude 30.67166667 deg, height 2010.344 m, pressure 798.4188 hPa,
*  water vapour 14.322 hPa, wavelength 0.532 um.
      CALL FCULZD_HPA(30.67166667D0, 2010.344D0, 798.4188D0, 14.322D0,
     .                0.532D0, ZTD, ZHD, ZWD)
      WRITE(*,'(A)') 'FCULZD_HPA header case  ZTD ZHD ZWD (m)'
      WRITE(*,'(3(1X,E24.17))') ZTD, ZHD, ZWD

*  The same site at two wavelengths, which is what makes the model dispersive: green
*  against infrared is ~11 cm of zenith delay, not the millimetre one might guess.
      CALL FCULZD_HPA(45D0, 100D0, 1013.25D0, 10D0, 0.532D0,ZTD,ZHD,ZWD)
      WRITE(*,'(A)') 'FCULZD_HPA 45 deg, 100 m, 532 nm'
      WRITE(*,'(3(1X,E24.17))') ZTD, ZHD, ZWD
      CALL FCULZD_HPA(45D0, 100D0, 1013.25D0, 10D0, 1.064D0,ZTD,ZHD,ZWD)
      WRITE(*,'(A)') 'FCULZD_HPA 45 deg, 100 m, 1064 nm'
      WRITE(*,'(3(1X,E24.17))') ZTD, ZHD, ZWD

*  FCUL_A, its own header case: the mapping function at 15 deg elevation.
      M = FCUL_A(30.67166667D0, 2075D0, 300.15D0, 15D0)
      WRITE(*,'(A)') 'FCUL_A header case'
      WRITE(*,'(1X,E24.17)') M

*  DEHANTTIDEINEL test case 1: 2009 April 13, 0 h.
      XSTA(1) = 4075578.385D0
      XSTA(2) =  931852.890D0
      XSTA(3) = 4801570.154D0
      XSUN(1) = 137859926952.015D0
      XSUN(2) = 54228127881.4350D0
      XSUN(3) = 23509422341.6960D0
      XMON(1) = -179996231.920342D0
      XMON(2) = -312468450.131567D0
      XMON(3) = -169288918.592160D0
      CALL DEHANTTIDEINEL(XSTA, 2009, 4, 13, 0D0, XSUN, XMON, DXTIDE)
      WRITE(*,'(A)') 'DEHANTTIDEINEL case 1  dx dy dz (m)'
      WRITE(*,'(3(1X,E24.17))') DXTIDE(1), DXTIDE(2), DXTIDE(3)

*  Test case 2: 2012 July 13, 0 h -- past the 2012-07-01 leap second, which moves
*  TAI-UTC to 35 s. Using 34 s there costs 5e-10 m, which is 500 times the tolerance
*  the port is held to and is exactly how that slip was caught.
      XSTA(1) =  1112189.660D0
      XSTA(2) = -4842955.026D0
      XSTA(3) =  3985352.284D0
      XSUN(1) = -54537460436.2357D0
      XSUN(2) =  130244288385.279D0
      XSUN(3) =  56463429031.5996D0
      XMON(1) =  300396716.912D0
      XMON(2) =  243238281.451D0
      XMON(3) =  120548075.939D0
      CALL DEHANTTIDEINEL(XSTA, 2012, 7, 13, 0D0, XSUN, XMON, DXTIDE)
      WRITE(*,'(A)') 'DEHANTTIDEINEL case 2  dx dy dz (m)'
      WRITE(*,'(3(1X,E24.17))') DXTIDE(1), DXTIDE(2), DXTIDE(3)

*  TDFRPH at 2009 June 25, 01h 10m 45s UTC -- the epoch HARDISP's own test case starts
*  from. HARDISP prints its displacements to six decimals, so the ocean-loading port is
*  pinned against them only to half a micron; these are the tidal arguments underneath at
*  full precision, so a future disagreement can be localised to the epoch chain or to the
*  admittance spline rather than merely observed.
      ITM(1) = 2009
      ITM(2) = 176
      ITM(3) = 1
      ITM(4) = 10
      ITM(5) = 45
      CALL TDFRPH_CASE(2, 0, 0, 0, 0, 0, 'M2 ')
      CALL TDFRPH_CASE(1, 1, 0, 0, 0, 0, 'K1 ')
      CALL TDFRPH_CASE(0, 0, 2, 0, 0, 0, 'Ssa')
      CALL TDFRPH_CASE(2, 2,-2, 0, 0, 0, 'S2 ')

*  SPLINE and EVAL on six unequally spaced knots. The ocean-loading admittance is only
*  ever splined through four knots per tidal species, and at four the index arithmetic
*  n-2 and n/2 agree, so a mutation of it is invisible there. Six knots separate them and
*  give the Rust port a reference for the general case.
*  SPLINE declares its dummy arrays a fixed 20 long and EVAL assigns to its own NN, so
*  the arrays have to be 20 and the count has to be a variable -- passing a literal there
*  writes to read-only memory and the program dies.
      NK = 6
      DO I = 1, NK
        XK(I) = XKD(I)
        UK(I) = UKD(I)
      ENDDO
      CALL SPLINE(NK, XK, UK, SK, AK)
      WRITE(*,'(A)') 'SPLINE 6 knots  second derivatives'
      WRITE(*,'(3(1X,E24.17))') SK(1), SK(2), SK(3)
      WRITE(*,'(3(1X,E24.17))') SK(4), SK(5), SK(6)
      WRITE(*,'(A)') 'EVAL between the knots'
      DO I = 1, NK - 1
        Y = 0.5*(XK(I) + XK(I+1))
        WRITE(*,'(1X,E24.17)') EVAL(Y, NK, XK, UK, SK)
      ENDDO
      END

*  One TDFRPH evaluation, printed. The COMMON block HARDISP shares with TDFRPH holds
*  (year, day of year, hour, minute, second) -- day of year, not month and day, which is
*  the one thing about the interface that is easy to get wrong. 2009 June 25 is day 176.
      SUBROUTINE TDFRPH_CASE(I1, I2, I3, I4, I5, I6, NAME)
      IMPLICIT NONE
      INTEGER I1, I2, I3, I4, I5, I6, IDOOD(6), ITM(5)
      DOUBLE PRECISION FREQ, PHASE
      CHARACTER*3 NAME
      COMMON/DATE/ITM
      IDOOD(1) = I1
      IDOOD(2) = I2
      IDOOD(3) = I3
      IDOOD(4) = I4
      IDOOD(5) = I5
      IDOOD(6) = I6
      CALL TDFRPH(IDOOD, FREQ, PHASE)
      WRITE(*,'(A,A,A)') 'TDFRPH ', NAME, '  freq cyc/day, phase deg'
      WRITE(*,'(2(1X,E24.17))') FREQ, PHASE
      END
