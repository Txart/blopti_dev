subroutine j_and_f(N, v, v_old, delta_t, delta_x, diri_bc, source, J, F)
! =====================================================
! Sets up Jacobian matrix for Newton-Rhapson method
! solution of the Backward Euler implicit finite diff
! 1D Bousinessq eq.
! Dirichlet in 0th position, no flux neumann in Nth position
! DUE TO RESTRICTIONS IN F2PY, THE SIZE OF THE OUTPUT ARRAY J
! CANNOT BE DYNAMICALLY ALLOCATED. I USE AN ASUMED ALLOCATION
! =====================================================
    
    integer, intent(in) :: N
    real, intent(in) :: delta_t, delta_x, diri_bc, source
    real, intent(in) :: v(N+1), v_old(N+1)
    real, intent(out) :: J(N+1,N+1), F(N+1)

    integer :: i
    real :: e


	! notation
    e = 1/(2*delta_x**2)

    do i=2,N
        J(i,i-1) = e*(-dif_prime(v(i-1))*v(i-1) - dif(v(i)) - dif(v(i-1)) + dif_prime(v(i-1))*v(i))
			
        J(i,i) = e*(-dif_prime(v(i))*v(i-1) + 2*dif_prime(v(i))*v(i) - dif_prime(v(i))*v(i+1) + dif(v(i+1)) & 
                        + 2*dif(v(i)) + dif(v(i-1))) + 1/delta_t
						
        J(i,i+1) = e*(dif_prime(v(i+1))*v(i) - dif(v(i+1)) - dif(v(i)) - dif_prime(v(i+1))*v(i+1))

	! F
	F(i) = -e*((dif(v(i)) + dif(v(i-1)))*v(i-1) -v(i)*(dif(v(i+1)) + 2*dif(v(i)) + dif(v(i-1))) &
                       + v(i+1)*(dif(v(i+1)) + dif(v(i)))) - source - v_old(i)/delta_t + v(i)/delta_t
			
    end do

    ! BC
    ! Diri in x=0
    J(1,1) = 1
    F(1) = diri_bc
    ! Neumann with diffusivity(u(L))*u'(L)=0 in x=N
    aL = dif(v(N))
    J(N+1,N+1) = e*(-dif_prime(v(N))*v(N) + 2*dif_prime(v(N+1))*v(N+1) + aL + 2*dif(v(N+1)) + dif(v(N))) + 1/delta_t
    J(N+1,N) = e*(-dif_prime(v(N))*v(N) + dif_prime(v(N))*v(N+1) - aL - 2*dif(v(N+1)) - dif(v(N))) -delta_x*e*(dif_prime(v(N+1)))
    F(N+1) = -e*((dif(v(N+1)) + dif(v(N)))*v(N) -v(N+1)*(aL + 2*dif(v(N+1)) + dif(v(N))) &
                      + v(N)*(aL + dif(v(N+1)))) - source - v_old(N+1)/delta_t + v(N+1)/delta_t
	
    return
end subroutine j_and_f

function dif(x) result(y)
    real, intent(in) :: x
    real :: y
	
	y = 1
	return
end function


subroutine dif_vector(x, x_length, D)
!===============================================
! Take in a vector of volumetric water contents,
! return the difussivity for each component
!===============================================

integer, intent(in) :: x_length
real, intent(inout) :: x(x_length) ! volumetric water content, theta
real, intent(out) :: D(x_length)

integer :: i

do i=1,x_length
    D(i) = dif(x(i))
end do

return
end subroutine

function dif_prime(x) result(y)
	real, intent(in) :: x
	real :: y
	
	y = 0.
	return
end function


subroutine dif_prime_vector(x, x_length, D_prime)
!===============================================
! Take in a vector of volumetric water contents,
! return the difussivity for each component
!===============================================
integer, intent(in) :: x_length
real, intent(inout) :: x(x_length) ! volumetric water content, theta
real, intent(out) :: D_prime(x_length)

integer :: i

do i=1,x_length
    D_prime(i) = dif_prime(x(i))
end do
return
end subroutine


!subroutine fd_fortran(v_ini, N, dt, dx, diri_bc, rel_tol, abs_tol, weight, max_internal_niter, timesteps)


!    return
!end subroutine fd_fortran

