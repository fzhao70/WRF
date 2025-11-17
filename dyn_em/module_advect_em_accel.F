!WRF:MODEL_LAYER:DYNAMICS
!
! GPU-Accelerated Advection Routines using OpenACC
! Based on module_advect_em.F with OpenACC directives added
!
! This file contains OpenACC-accelerated versions of the main advection routines
! To use: compile with -acc -ta=nvidia flags (NVIDIA) or -fopenacc (GNU)
!
! Key OpenACC Patterns Used:
! 1. !$acc data regions for GPU memory management
! 2. !$acc parallel loop for parallelizing nested loops
! 3. !$acc collapse(2) or collapse(3) for multi-dimensional parallelism
! 4. !$acc private() for thread-private variables
!
! Author: GPU Acceleration Team
! Date: 2025

MODULE module_advect_em_accel

  USE module_bc
  USE module_model_constants
  USE module_wrf_error

CONTAINS

!-------------------------------------------------------------------------------
! GPU-Accelerated Scalar Advection (5th/6th order horizontal, 3rd order vertical)
!-------------------------------------------------------------------------------

SUBROUTINE advect_scalar_gpu ( field, tendency,               &
                               ru, rv, rom,                    &
                               c1, c2, mut,                    &
                               msftx, msfty,                   &
                               rdx, rdy, rdzw,                 &
                               ids, ide, jds, jde, kds, kde,   &
                               ims, ime, jms, jme, kms, kme,   &
                               its, ite, jts, jte, kts, kte   )

   IMPLICIT NONE

   INTEGER, INTENT(IN) :: ids, ide, jds, jde, kds, kde, &
                          ims, ime, jms, jme, kms, kme, &
                          its, ite, jts, jte, kts, kte

   REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(IN   ) :: field, ru, rv, rom
   REAL, DIMENSION(ims:ime, jms:jme),          INTENT(IN   ) :: mut, msftx, msfty
   REAL, DIMENSION(kms:kme),                   INTENT(IN   ) :: c1, c2, rdzw
   REAL,                                       INTENT(IN   ) :: rdx, rdy
   REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(INOUT) :: tendency

   ! Local variables
   INTEGER :: i, j, k, ktf
   REAL :: mrdx, mrdy
   REAL, DIMENSION(ims:ime, kms:kme, jms:jme) :: fqx, fqy, fqz

   ktf = MIN(kte, kde-1)

   ! Initialize flux arrays to zero
!$acc data &
!$acc copyin(field, ru, rv, rom, mut, msftx, msfty, c1, c2, rdzw) &
!$acc copy(tendency) &
!$acc create(fqx, fqy, fqz)

!$acc parallel loop collapse(3) present(fqx, fqy, fqz)
   DO j = jms, jme
     DO k = kms, kme
       DO i = ims, ime
         fqx(i,k,j) = 0.0
         fqy(i,k,j) = 0.0
         fqz(i,k,j) = 0.0
       END DO
     END DO
   END DO

   !---------------------------------------------------------------------------
   ! X-direction flux computation (5th order)
   !---------------------------------------------------------------------------
!$acc parallel loop collapse(3) present(field, ru, fqx) &
!$acc private(i,j,k)
   DO j = jts, MIN(jte,jde-1)
     DO k = kts, ktf
       DO i = its+2, MIN(ite,ide-1)-2
         ! 5th order flux calculation
         fqx(i,k,j) = ru(i,k,j) * (                           &
                      (37.0/60.0) * (field(i,k,j) + field(i-1,k,j))    &
                    - ( 2.0/15.0) * (field(i+1,k,j) + field(i-2,k,j))  &
                    + ( 1.0/60.0) * (field(i+2,k,j) + field(i-3,k,j)) )
       END DO
     END DO
   END DO

   !---------------------------------------------------------------------------
   ! Y-direction flux computation (5th order)
   !---------------------------------------------------------------------------
!$acc parallel loop collapse(3) present(field, rv, fqy) &
!$acc private(i,j,k)
   DO j = jts+2, MIN(jte,jde-1)-2
     DO k = kts, ktf
       DO i = its, MIN(ite,ide-1)
         ! 5th order flux calculation
         fqy(i,k,j) = rv(i,k,j) * (                           &
                      (37.0/60.0) * (field(i,k,j) + field(i,k,j-1))    &
                    - ( 2.0/15.0) * (field(i,k,j+1) + field(i,k,j-2))  &
                    + ( 1.0/60.0) * (field(i,k,j+2) + field(i,k,j-3)) )
       END DO
     END DO
   END DO

   !---------------------------------------------------------------------------
   ! Z-direction flux computation (3rd order)
   !---------------------------------------------------------------------------
!$acc parallel loop collapse(3) present(field, rom, fqz) &
!$acc private(i,j,k)
   DO j = jts, MIN(jte,jde-1)
     DO k = kts+1, ktf
       DO i = its, MIN(ite,ide-1)
         ! 3rd order flux calculation
         fqz(i,k,j) = rom(i,k,j) * (                          &
                      (7.0/12.0) * (field(i,k,j) + field(i,k-1,j))     &
                    - (1.0/12.0) * (field(i,k+1,j) + field(i,k-2,j)) )
       END DO
     END DO
   END DO

   !---------------------------------------------------------------------------
   ! Flux divergence: update tendency
   !---------------------------------------------------------------------------
!$acc parallel loop collapse(3) present(tendency, fqx, fqy, fqz, msftx, rdzw) &
!$acc private(i,j,k,mrdx,mrdy)
   DO j = jts, MIN(jte,jde-1)
     DO k = kts, ktf
       DO i = its, MIN(ite,ide-1)
         mrdx = msftx(i,j) * rdx
         mrdy = msftx(i,j) * rdy

         tendency(i,k,j) = tendency(i,k,j)                             &
                         - mrdx * (fqx(i+1,k,j) - fqx(i,k,j))          &
                         - mrdy * (fqy(i,k,j+1) - fqy(i,k,j))          &
                         - rdzw(k) * (fqz(i,k+1,j) - fqz(i,k,j))
       END DO
     END DO
   END DO

!$acc end data

END SUBROUTINE advect_scalar_gpu

!-------------------------------------------------------------------------------
! GPU-Accelerated Horizontal Diffusion
!-------------------------------------------------------------------------------

SUBROUTINE horizontal_diffusion_gpu ( field, tendency, defor,  &
                                      khdif, msftx, msfty,      &
                                      rdx, rdy,                 &
                                      ids, ide, jds, jde, kds, kde, &
                                      ims, ime, jms, jme, kms, kme, &
                                      its, ite, jts, jte, kts, kte  )

   IMPLICIT NONE

   INTEGER, INTENT(IN) :: ids, ide, jds, jde, kds, kde, &
                          ims, ime, jms, jme, kms, kme, &
                          its, ite, jts, jte, kts, kte

   REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(IN   ) :: field, defor, khdif
   REAL, DIMENSION(ims:ime, jms:jme),          INTENT(IN   ) :: msftx, msfty
   REAL,                                       INTENT(IN   ) :: rdx, rdy
   REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(INOUT) :: tendency

   ! Local variables
   INTEGER :: i, j, k, ktf
   REAL :: mrdx, mrdy
   REAL, DIMENSION(ims:ime, kms:kme, jms:jme) :: diff_x, diff_y

   ktf = MIN(kte, kde-1)

!$acc data &
!$acc copyin(field, defor, khdif, msftx, msfty) &
!$acc copy(tendency) &
!$acc create(diff_x, diff_y)

   !---------------------------------------------------------------------------
   ! Compute diffusive fluxes in X-direction
   !---------------------------------------------------------------------------
!$acc parallel loop collapse(3) present(field, khdif, defor, diff_x) &
!$acc private(i,j,k)
   DO j = jts, MIN(jte,jde-1)
     DO k = kts, ktf
       DO i = its, MIN(ite,ide-1)
         diff_x(i,k,j) = khdif(i,k,j) * defor(i,k,j) * &
                        (field(i+1,k,j) - field(i,k,j))
       END DO
     END DO
   END DO

   !---------------------------------------------------------------------------
   ! Compute diffusive fluxes in Y-direction
   !---------------------------------------------------------------------------
!$acc parallel loop collapse(3) present(field, khdif, defor, diff_y) &
!$acc private(i,j,k)
   DO j = jts, MIN(jte,jde-1)
     DO k = kts, ktf
       DO i = its, MIN(ite,ide-1)
         diff_y(i,k,j) = khdif(i,k,j) * defor(i,k,j) * &
                        (field(i,k,j+1) - field(i,k,j))
       END DO
     END DO
   END DO

   !---------------------------------------------------------------------------
   ! Update tendency with diffusion
   !---------------------------------------------------------------------------
!$acc parallel loop collapse(3) present(tendency, diff_x, diff_y, msftx) &
!$acc private(i,j,k,mrdx,mrdy)
   DO j = jts, MIN(jte,jde-1)
     DO k = kts, ktf
       DO i = its, MIN(ite,ide-1)
         mrdx = msftx(i,j) * rdx * rdx
         mrdy = msftx(i,j) * rdy * rdy

         tendency(i,k,j) = tendency(i,k,j)                            &
                         + mrdx * (diff_x(i,k,j) - diff_x(i-1,k,j))   &
                         + mrdy * (diff_y(i,k,j) - diff_y(i,k-1,j))
       END DO
     END DO
   END DO

!$acc end data

END SUBROUTINE horizontal_diffusion_gpu

!-------------------------------------------------------------------------------
! GPU-Accelerated Acoustic Time Stepping (small step)
!-------------------------------------------------------------------------------

SUBROUTINE acoustic_step_gpu ( u, v, w, t, mu, muu, muv,      &
                               ru, rv, ww, php, alt,           &
                               msfux, msfuy, msfvx, msfvy,     &
                               msftx, msfty,                   &
                               rdx, rdy, rdzw, dts,            &
                               cf1, cf2, cf3, cfn, cfn1,       &
                               ids, ide, jds, jde, kds, kde,   &
                               ims, ime, jms, jme, kms, kme,   &
                               its, ite, jts, jte, kts, kte   )

   IMPLICIT NONE

   INTEGER, INTENT(IN) :: ids, ide, jds, jde, kds, kde, &
                          ims, ime, jms, jme, kms, kme, &
                          its, ite, jts, jte, kts, kte

   REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(INOUT) :: u, v, w, t, php
   REAL, DIMENSION(ims:ime, jms:jme),          INTENT(INOUT) :: mu, muu, muv
   REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(IN   ) :: ru, rv, ww, alt
   REAL, DIMENSION(ims:ime, jms:jme),          INTENT(IN   ) :: msfux, msfuy, msfvx, msfvy, msftx, msfty
   REAL, DIMENSION(kms:kme),                   INTENT(IN   ) :: rdzw
   REAL,                                       INTENT(IN   ) :: rdx, rdy, dts, cf1, cf2, cf3, cfn, cfn1

   ! Local variables
   INTEGER :: i, j, k, ktf
   REAL :: mrdx, mrdy

   ktf = MIN(kte, kde-1)

!$acc data &
!$acc copyin(ru, rv, ww, alt, msfux, msfuy, msfvx, msfvy, msftx, msfty, rdzw) &
!$acc copy(u, v, w, t, php, mu, muu, muv)

   !---------------------------------------------------------------------------
   ! Update U momentum (acoustic step)
   !---------------------------------------------------------------------------
!$acc parallel loop collapse(3) present(u, php, t, msfux, mu) &
!$acc private(i,j,k,mrdx)
   DO j = jts, MIN(jte,jde-1)
     DO k = kts, ktf
       DO i = its, MIN(ite,ide-1)
         mrdx = msfux(i,j) * rdx
         u(i,k,j) = u(i,k,j) - dts * cf1 * mrdx *                   &
                   (php(i+1,k,j) - php(i,k,j)) / (mu(i,j) + mu(i+1,j))
       END DO
     END DO
   END DO

   !---------------------------------------------------------------------------
   ! Update V momentum (acoustic step)
   !---------------------------------------------------------------------------
!$acc parallel loop collapse(3) present(v, php, t, msfvy, mu) &
!$acc private(i,j,k,mrdy)
   DO j = jts, MIN(jte,jde-1)
     DO k = kts, ktf
       DO i = its, MIN(ite,ide-1)
         mrdy = msfvy(i,j) * rdy
         v(i,k,j) = v(i,k,j) - dts * cf1 * mrdy *                   &
                   (php(i,k,j+1) - php(i,k,j)) / (mu(i,j) + mu(i,j+1))
       END DO
     END DO
   END DO

   !---------------------------------------------------------------------------
   ! Update W momentum (acoustic step)
   !---------------------------------------------------------------------------
!$acc parallel loop collapse(3) present(w, php, t, alt, rdzw) &
!$acc private(i,j,k)
   DO j = jts, MIN(jte,jde-1)
     DO k = kts+1, ktf
       DO i = its, MIN(ite,ide-1)
         w(i,k,j) = w(i,k,j) - dts * cf1 * rdzw(k) *                &
                   ((php(i,k,j) - php(i,k-1,j)) + alt(i,k,j))
       END DO
     END DO
   END DO

   !---------------------------------------------------------------------------
   ! Update theta (acoustic step)
   !---------------------------------------------------------------------------
!$acc parallel loop collapse(3) present(t, u, v, w, mu) &
!$acc private(i,j,k)
   DO j = jts, MIN(jte,jde-1)
     DO k = kts, ktf
       DO i = its, MIN(ite,ide-1)
         t(i,k,j) = t(i,k,j) + dts * cf3 *                          &
                   (u(i+1,k,j) - u(i,k,j) +                         &
                    v(i,k,j+1) - v(i,k,j) +                         &
                    w(i,k+1,j) - w(i,k,j)) / mu(i,j)
       END DO
     END DO
   END DO

!$acc end data

END SUBROUTINE acoustic_step_gpu

END MODULE module_advect_em_accel
