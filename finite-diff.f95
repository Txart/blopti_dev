subroutine finite_diff(v, v_old, N, dt, dx, source, diri_bc, rel_tol, abs_tolerance, weight, max_internal_niter, v_sol)
! =====================================================
! Finite differences solution algorithm
! Uses lapack library solvers
! =====================================================
	
	
	integer, intent(in) :: max_internal_niter, N
	real, intent(in) :: dt, dx, diri_bc, rel_tol, abs_tolerance, weight, source
	real, intent(in) :: v_old(N+1)
	real, intent(in):: v(N+1)
	real, intent(out) :: v_sol(N+1)

	
	integer :: i, info, ipiv(N+1)
	real :: residue, normF
	real :: d(N+1), dl(N), du(N), du2(N-1), eps_x(N+1), efe(N+1)
	
	du2 = 0.0

	v_sol = v

	do i=1,max_internal_niter
		call j_diag_parts_and_f(N, v_sol, v_old, diri_bc, source, dx, dt, d, du, dl, efe)
		call sgttrf(N+1, dl, d, du, ipiv, info) ! LU decomposition needed for solving
		! if (info<0) then
			! print *, "some parameter  in the matrix has an illegal value"
		! else if (info>0)
			! print *, "U is exactly singular"
		eps_x = -efe ! eps_x gets rewritten with the solution
		call sgttrs('N', N+1, N+1, dl, d, du, du2, ipiv, eps_x, N+1, info) ! solve with Lapack
		v_sol = v_sol + weight*eps_x
		
        ! stopping criterion
        residue =  sqrt ( sum ( efe(:N+1)*efe(:N+1) )) - rel_tol
        if (residue < abs_tolerance) then
			print *, 'Solution of the Newton linear system in {i} iterations'
			exit
		end if
	end do
	
	return
end subroutine finite_diff

subroutine j_diag_parts_and_f(N, v, v_old, diri_bc, source, delta_x, delta_t, jdi, jsuperdi, jsubdi, F)
!========================================
!diagonal, sub and super diag elements of jacobian mnatrix J
! Returns also F
! Needed to solve using LAPACK tridiagonal
! solvers. Effectively, same as J and F below.
!jdi = J diagonal; also sub and superdiagonals
!========================================
    integer, intent(in) :: N
    real, intent(in) :: delta_t, delta_x
    real, intent(in) :: v(N+1), v_old(N+1)
    real, intent(out) :: jdi(N+1), jsuperdi(N), jsubdi(N), F(N+1)

	integer :: i
	real :: e
	
	! notation
    e = 1/(2*delta_x**2)
	
	do i=2,N
		jsubdi(i) = e*(-dif_prime(v(i-1))*v(i-1) - dif(v(i)) - dif(v(i-1)) + dif_prime(v(i-1))*v(i))
		jdi(i) = e*(-dif_prime(v(i))*v(i-1) + 2*dif_prime(v(i))*v(i) - dif_prime(v(i))*v(i+1) + dif(v(i+1)) & 
                        + 2*dif(v(i)) + dif(v(i-1))) + 1/delta_t
		jsuperdi = e*(dif_prime(v(i+1))*v(i) - dif(v(i+1)) - dif(v(i)) - dif_prime(v(i+1))*v(i+1))
		F(i) = -e*((dif(v(i)) + dif(v(i-1)))*v(i-1) -v(i)*(dif(v(i+1)) + 2*dif(v(i)) + dif(v(i-1))) &
                       + v(i+1)*(dif(v(i+1)) + dif(v(i)))) - source - v_old(i)/delta_t + v(i)/delta_t
		
	end do

	! diri BC in x=0
	jdi(1) =1
	F(1) = diri_bc
	!Neuman BC
	jdi(N+1) = e*(-dif_prime(v(N))*v(N) + 2*dif_prime(v(N+1))*v(N+1) + dif(v(N)) + 2*dif(v(N+1)) + dif(v(N))) + 1/delta_t
	jsubdi(N) = e*(-dif_prime(v(N))*v(N) + dif_prime(v(N))*v(N+1) - dif(v(N)) - 2*dif(v(N+1)) - dif(v(N))) &
					-delta_x*e*(dif_prime(v(N+1)))
	F(N+1) = -e*((dif(v(N+1)) + dif(v(N)))*v(N) -v(N+1)*(dif(v(N)) + 2*dif(v(N+1)) + dif(v(N))) &
                      + v(N)*(dif(v(N)) + dif(v(N+1)))) - source - v_old(N+1)/delta_t + v(N+1)/delta_t

	return

end subroutine j_diag_parts_and_f

! subroutine only_f(N, v, v_old, delta_t, delta_x, diri_bc, source, F): ! NOT IN USE RIGHT NOW
! !==============================================================
! ! Only compute F, not J and F as below. Code is identical
! !==============================================================
	! integer, intent(in) :: N
    ! real, intent(in) :: delta_t, delta_x, diri_bc, source
    ! real, intent(in) :: v(N+1), v_old(N+1)
    ! real, intent(out) :: F(N+1)

    ! integer :: i
	
	! do i=2,N
        ! F(i) = -1/(2*delta_x**2)*((dif(v(i)) + dif(v(i-1)))*v(i-1) -v(i)*(dif(v(i+1)) + 2*dif(v(i)) + dif(v(i-1))) &
                       ! + v(i+1)*(dif(v(i+1)) + dif(v(i)))) - source - v_old(i)/delta_t + v(i)/delta_t
	! end do
	
	! !BC
	! F(1) = diri_bc
	! F(N+1) = -1/(2*delta_x**2)*((dif(v(N+1)) + dif(v(N)))*v(N) -v(N+1)*(dif(v(N)) + 2*dif(v(N+1)) + dif(v(N))) &
                      ! + v(N)*(dif(v(N)) + dif(v(N+1)))) - source - v_old(N+1)/delta_t + v(N+1)/delta_t
	

! end subroutine only_f

function dif(x, bi, s1, s2, t1, t2) result(y)
			real, intent(in) :: x, bi, s1, s2, t1, t2
			real :: y, A
			
			! notation
			A = s2 * exp(-s1)*x + exp(s2*bi)

			y = exp(t1-s1)/t2 * (A**(t2/s2) - exp(t2*bi))/A
			return
		end function

		function dif_prime(x, bi, s1, s2, t1, t2) result(y)
			real, intent(in) :: x, bi, s1, s2, t1, t2
			real :: y, A
			
			! notation
			A = s2 * exp(-s1)*x + exp(s2*bi)
			
			y = exp(t1-2*s1)*s2/(t2*A**2) * (A**(t2/s2)*(t2/s2-1) + exp(t2*bi))
			return
		end function

subroutine j_and_f(N, v, v_old, b, delta_t, delta_x, diri_bc, s1, s2, t1, t2, source, J, F)
! =====================================================
! Sets up Jacobian matrix for Newton-Rhapson method
! solution of the Backward Euler implicit finite diff
! 1D Bousinessq eq.
! Dirichlet in 0th position, no flux neumann in Nth position
! DUE TO RESTRICTIONS IN F2PY, THE SIZE OF THE OUTPUT ARRAY J
! CANNOT BE DYNAMICALLY ALLOCATED. I USE AN ASUMED ALLOCATION
! =====================================================
    
    integer, intent(in) :: N
    real, intent(in) :: delta_t, delta_x, diri_bc, source, s1, s2, t1, t2
    real, intent(in) :: v(N+1), v_old(N+1), b(N+1)
    real, intent(out) :: J(N+1,N+1), F(N+1)

    integer :: i
    real :: e	

    ! notation
    e = 1/(2*delta_x**2)

    do i=2,N
        J(i,i-1) = e*(-dif_prime(v(i-1), b(i-1), s1, s2, t1, t2)*v(i-1) - dif(v(i), b(i), s1, s2, t1, t2) &
					- dif(v(i-1), b(i-1), s1, s2, t1, t2) + dif_prime(v(i-1), b(i-1), s1, s2, t1, t2)*v(i))

        J(i,i) = e*(-dif_prime(v(i), b(i), s1, s2, t1, t2)*v(i-1) + 2*dif_prime(v(i), b(i), s1, s2, t1, t2)*v(i) - & 
					dif_prime(v(i), b(i), s1, s2, t1, t2)*v(i+1) + dif(v(i+1), b(i+1), s1, s2, t1, t2) & 
                        + 2*dif(v(i), b(i), s1, s2, t1, t2) + dif(v(i-1), b(i-1), s1, s2, t1, t2)) + 1/delta_t

        J(i,i+1) = e*(dif_prime(v(i+1), b(i+1), s1, s2, t1, t2)*v(i) - dif(v(i+1), b(i+1), s1, s2, t1, t2) &
					- dif(v(i), b(i), s1, s2, t1, t2) - dif_prime(v(i+1), b(i+1), s1, s2, t1, t2)*v(i+1))

        ! F
        F(i) = -e*((dif(v(i), b(i), s1, s2, t1, t2) + dif(v(i-1), b(i-1), s1, s2, t1, t2))*v(i-1) & 
				-v(i)*(dif(v(i+1), b(i+1), s1, s2, t1, t2) & 
					+ 2*dif(v(i), b(i), s1, s2, t1, t2) + dif(v(i-1), b(i-1), s1, s2, t1, t2)) &
                       + v(i+1)*(dif(v(i+1), b(i+1), s1, s2, t1, t2) + dif(v(i), b(i), s1, s2, t1, t2))) &
					   - source - v_old(i)/delta_t + v(i)/delta_t

    end do

    ! BC
    ! Diri in x=0
    J(1,1) = 1
    F(1) = diri_bc
    ! Neumann with diffusivity(u(L))*u'(L)=0 in x=N
    aL = dif(v(N), b(N), s1, s2, t1, t2)
    J(N+1,N+1) = e*(-dif_prime(v(N), b(N), s1, s2, t1, t2)*v(N) + 2*dif_prime(v(N+1), b(N+1), s1, s2, t1, t2)*v(N+1) &
					+ aL + 2*dif(v(N+1), b(N+1), s1, s2, t1, t2) + dif(v(N), b(N), s1, s2, t1, t2)) + 1/delta_t
    J(N+1,N) = e*(-dif_prime(v(N), b(N), s1, s2, t1, t2)*v(N) + dif_prime(v(N), b(N), s1, s2, t1, t2)*v(N+1) - aL &
					- 2*dif(v(N+1), b(N+1), s1, s2, t1, t2) - dif(v(N), b(N), s1, s2, t1, t2)) &
					-delta_x*e*(dif_prime(v(N+1), b(N+1), s1, s2, t1, t2))
    F(N+1) = -e*((dif(v(N+1), b(N+1), s1, s2, t1, t2) + dif(v(N), b(N), s1, s2, t1, t2))*v(N) & 
					-v(N+1)*(aL + 2*dif(v(N+1), b(N+1), s1, s2, t1, t2) + & 
						dif(v(N), b(N), s1, s2, t1, t2))+ v(N)*(aL + dif(v(N+1), b(N+1), s1, s2, t1, t2))) &
						- source - v_old(N+1)/delta_t + v(N+1)/delta_t
		

end subroutine j_and_f


! subroutine dif_vector(x, x_length, D)
! !===============================================
! ! Take in a vector of volumetric water contents,
! ! return the difussivity for each component
! !===============================================

! integer, intent(in) :: x_length
! real, intent(inout) :: x(x_length) ! volumetric water content, theta
! real, intent(out) :: D(x_length)

! integer :: i

! do i=1,x_length
    ! D(i) = dif(x(i))
! end do

! return
! end subroutine




! subroutine dif_prime_vector(x, x_length, D_prime)
! !===============================================
! ! Take in a vector of volumetric water contents,
! ! return the difussivity for each component
! !===============================================
! integer, intent(in) :: x_length
! real, intent(inout) :: x(x_length) ! volumetric water content, theta
! real, intent(out) :: D_prime(x_length)

! integer :: i

! do i=1,x_length
    ! D_prime(i) = dif_prime(x(i))
! end do
! return
! end subroutine




