subroutine j_and_f(N, v, v_old, b, delta_t, delta_x, s1, s2, t1, t2, source, J, F)
! =====================================================
! Sets up Jacobian matrix for Newton-Rhapson method
! solution of the Backward Euler implicit finite diff
! 1D Bousinessq eq.
! Dirichlet in 0th position, no-flux neumann in Nth position.
! Diri BC must be present in the initial vector v
! DUE TO RESTRICTIONS IN F2PY, THE SIZE OF THE OUTPUT ARRAY J
! CANNOT BE DYNAMICALLY ALLOCATED. I USE AN ASUMED ALLOCATION
! =====================================================
    
    integer, intent(in) :: N
    real, intent(in) :: delta_t, delta_x, source, s1, s2, t1, t2
    real, intent(in) :: v(N+1), v_old(N+1), b(N+1)
    real, intent(out) :: J(N+1,N+1), F(N+1)

    integer :: i
    real :: e	

    ! notation
    e = 1/(2*delta_x**2)

    do i=2,N
        J(i,i-1) = e*(-dif_prime(v(i-1), b(i-1))*v(i-1) - dif(v(i), b(i)) - dif(v(i-1), b(i-1)) + & 
						dif_prime(v(i-1), b(i-1))*v(i))

        J(i,i) = e*(-dif_prime(v(i), b(i))*v(i-1) + 2*dif_prime(v(i), b(i))*v(i) - & 
					dif_prime(v(i), b(i))*v(i+1) + dif(v(i+1), b(i+1)) & 
                        + 2*dif(v(i), b(i)) + dif(v(i-1), b(i-1))) + 1/delta_t

        J(i,i+1) = e*(dif_prime(v(i+1), b(i+1))*v(i) - dif(v(i+1), b(i+1)) - dif(v(i), b(i)) &
						- dif_prime(v(i+1), b(i+1))*v(i+1))

        ! F
        F(i) = -e*((dif(v(i), b(i)) + dif(v(i-1), b(i-1)))*v(i-1) -v(i)*(dif(v(i+1), b(i+1)) & 
							+ 2*dif(v(i), b(i)) + dif(v(i-1), b(i-1))) &
                       + v(i+1)*(dif(v(i+1), b(i+1)) + dif(v(i), b(i)))) - source - v_old(i)/delta_t + v(i)/delta_t

    end do

    ! BC
    ! Diri in x=0
    J(1,1) = 1
    F(1) = 0.0
    ! Neumann with diffusivity(u(L))*u'(L)=0 in x=N
    aL = dif(v(N), b(N))
    J(N+1,N+1) = e*(-dif_prime(v(N), b(N))*v(N) + 2*dif_prime(v(N+1), b(N+1))*v(N+1) &
					+ aL + 2*dif(v(N+1), b(N+1)) + dif(v(N), b(N))) + 1/delta_t
    J(N+1,N) = e*(-dif_prime(v(N), b(N))*v(N) + dif_prime(v(N), b(N))*v(N+1) - aL &
					- 2*dif(v(N+1), b(N+1)) - dif(v(N), b(N))) -delta_x*e*(dif_prime(v(N+1), b(N+1)))
    F(N+1) = -e*((dif(v(N+1), b(N+1)) + dif(v(N), b(N)))*v(N) -v(N+1)*(aL + 2*dif(v(N+1), b(N+1)) + & 
						dif(v(N), b(N)))+ v(N)*(aL + dif(v(N+1), b(N+1)))) &
						- source - v_old(N+1)/delta_t + v(N+1)/delta_t

	contains ! Add subroutines in here in order to share parameters
		real function dif(x,bi)
			real, intent(in) :: x, bi
			real :: A
			
			! notation
			A = s2 * exp(-s1)*x + exp(s2*bi)

			dif = exp(t1-s1)/t2 * (A**(t2/s2) - exp(t2*bi))/A
		end function dif

		real function dif_prime(x, bi)
			real, intent(in) :: x, bi
			real :: A
			
			! notation
			A = s2 * exp(-s1)*x + exp(s2*bi)

			dif_prime = exp(t1-2*s1)*s2/(t2*A**2) * (A**(t2/s2)*(t2/s2-1) + exp(t2*bi))
		end function dif_prime

end subroutine j_and_f


subroutine j_and_f_diri_both(N, v, v_old, b, delta_t, delta_x, s1, s2, t1, t2, source, J, F)
! =====================================================
! Sets up Jacobian matrix for Newton-Rhapson method
! solution of the Backward Euler implicit finite diff
! 1D Bousinessq eq.
! Dirichlet in 0th and Nth position
! Diri BCs must be present in the input vector v.
! DUE TO RESTRICTIONS IN F2PY, THE SIZE OF THE OUTPUT ARRAY J
! CANNOT BE DYNAMICALLY ALLOCATED. I USE AN ASUMED ALLOCATION
! =====================================================
    
    integer, intent(in) :: N
    real, intent(in) :: delta_t, delta_x, source, s1, s2, t1, t2
    real, intent(in) :: v(N+1), v_old(N+1), b(N+1)
    real, intent(out) :: J(N+1,N+1), F(N+1)

    integer :: i
    real :: e	

    ! notation
    e = 1/(2*delta_x**2)

    do i=2,N
        J(i,i-1) = e*(-dif_prime(v(i-1), b(i-1))*v(i-1) - dif(v(i), b(i)) - dif(v(i-1), b(i-1)) + & 
						dif_prime(v(i-1), b(i-1))*v(i))

        J(i,i) = e*(-dif_prime(v(i), b(i))*v(i-1) + 2*dif_prime(v(i), b(i))*v(i) - & 
					dif_prime(v(i), b(i))*v(i+1) + dif(v(i+1), b(i+1)) & 
                        + 2*dif(v(i), b(i)) + dif(v(i-1), b(i-1))) + 1/delta_t

        J(i,i+1) = e*(dif_prime(v(i+1), b(i+1))*v(i) - dif(v(i+1), b(i+1)) - dif(v(i), b(i)) &
						- dif_prime(v(i+1), b(i+1))*v(i+1))

        ! F
        F(i) = -e*((dif(v(i), b(i)) + dif(v(i-1), b(i-1)))*v(i-1) -v(i)*(dif(v(i+1), b(i+1)) & 
							+ 2*dif(v(i), b(i)) + dif(v(i-1), b(i-1))) &
                       + v(i+1)*(dif(v(i+1), b(i+1)) + dif(v(i), b(i)))) - source - v_old(i)/delta_t + v(i)/delta_t

    end do

    ! BC
    ! Diri in x=0
    J(1,1) = 1.0
    F(1) = 0.0
    ! Diri in x=N
    aL = dif(v(N), b(N))
    J(N+1,N+1) = 1.0
    F(N+1) = 0.0

	contains ! Add subroutines in here in order to share parameters
		real function dif(x,bi)
			real, intent(in) :: x, bi
			real :: A
			
			! notation
			A = s2 * exp(-s1)*x + exp(s2*bi)

			dif = exp(t1-s1)/t2 * (A**(t2/s2) - exp(t2*bi))/A
		end function dif

		real function dif_prime(x, bi)
			real, intent(in) :: x, bi
			real :: A
			
			! notation
			A = s2 * exp(-s1)*x + exp(s2*bi)

			dif_prime = exp(t1-2*s1)*s2/(t2*A**2) * (A**(t2/s2)*(t2/s2-1) + exp(t2*bi))
		end function dif_prime

end subroutine j_and_f