      module hex8r_generated_wrappers
      implicit none

      contains

      subroutine pack_2d_rowmajor(mat, n1, n2, vec)
      implicit none
      integer n1, n2, i, j, idx
      real*8 mat(n1,n2), vec(n1*n2)

      idx = 0
      do i = 1, n1
         do j = 1, n2
            idx = idx + 1
            vec(idx) = mat(i,j)
         enddo
      enddo
      end subroutine pack_2d_rowmajor

      subroutine unpack_2d_rowmajor(vec, n1, n2, mat)
      implicit none
      integer n1, n2, i, j, idx
      real*8 vec(n1*n2), mat(n1,n2)

      idx = 0
      do i = 1, n1
         do j = 1, n2
            idx = idx + 1
            mat(i,j) = vec(idx)
         enddo
      enddo
      end subroutine unpack_2d_rowmajor

      subroutine unpack_3d_rowmajor(vec, n1, n2, n3, arr)
      implicit none
      integer n1, n2, n3, i, j, k, idx
      real*8 vec(n1*n2*n3), arr(n1,n2,n3)

      idx = 0
      do i = 1, n1
         do j = 1, n2
            do k = 1, n3
               idx = idx + 1
               arr(i,j,k) = vec(idx)
            enddo
         enddo
      enddo
      end subroutine unpack_3d_rowmajor

      subroutine unpack_4d_rowmajor(vec, n1, n2, n3, n4, arr)
      implicit none
      integer n1, n2, n3, n4, i, j, k, l, idx
      real*8 vec(n1*n2*n3*n4), arr(n1,n2,n3,n4)

      idx = 0
      do i = 1, n1
         do j = 1, n2
            do k = 1, n3
               do l = 1, n4
                  idx = idx + 1
                  arr(i,j,k,l) = vec(idx)
               enddo
            enddo
         enddo
      enddo
      end subroutine unpack_4d_rowmajor

      subroutine hex8r_op_bbar_grad_wrapper(COORD, BiI, VOL)
      implicit none
      real*8 COORD(8,3), BiI(8,3), VOL
      real*8 in_vec(24), out_vec(25)
      integer i

      do i = 1, 8
         in_vec((i-1)*3 + 1) = COORD(i,1)
         in_vec((i-1)*3 + 2) = COORD(i,2)
         in_vec((i-1)*3 + 3) = COORD(i,3)
      enddo

      call compute_hex8r_op_bbar_grad(in_vec, out_vec)

      do i = 1, 8
         BiI(i,1) = out_vec((i-1)*3 + 1)
         BiI(i,2) = out_vec((i-1)*3 + 2)
         BiI(i,3) = out_vec((i-1)*3 + 3)
      enddo
      VOL = out_vec(25)
      end subroutine hex8r_op_bbar_grad_wrapper

      subroutine hex8r_op_jacobian_center_wrapper(COORD, J, detJ,
     1 Jinv)
      implicit none
      real*8 COORD(8,3), J(3,3), detJ, Jinv(3,3)
      real*8 in_vec(24), out_vec(19)
      integer i

      do i = 1, 8
         in_vec((i-1)*3 + 1) = COORD(i,1)
         in_vec((i-1)*3 + 2) = COORD(i,2)
         in_vec((i-1)*3 + 3) = COORD(i,3)
      enddo

      call compute_hex8r_op_jacobian_center(in_vec, out_vec)

      J(1,1) = out_vec(1)
      J(1,2) = out_vec(2)
      J(1,3) = out_vec(3)
      J(2,1) = out_vec(4)
      J(2,2) = out_vec(5)
      J(2,3) = out_vec(6)
      J(3,1) = out_vec(7)
      J(3,2) = out_vec(8)
      J(3,3) = out_vec(9)
      detJ = out_vec(10)
      Jinv(1,1) = out_vec(11)
      Jinv(1,2) = out_vec(12)
      Jinv(1,3) = out_vec(13)
      Jinv(2,1) = out_vec(14)
      Jinv(2,2) = out_vec(15)
      Jinv(2,3) = out_vec(16)
      Jinv(3,1) = out_vec(17)
      Jinv(3,2) = out_vec(18)
      Jinv(3,3) = out_vec(19)
      end subroutine hex8r_op_jacobian_center_wrapper

      subroutine hex8r_op_form_B_wrapper(BiI, B)
      implicit none
      real*8 BiI(8,3), B(6,24)
      real*8 in_vec(24), out_vec(144)
      integer i

      do i = 1, 8
         in_vec((i-1)*3 + 1) = BiI(i,1)
         in_vec((i-1)*3 + 2) = BiI(i,2)
         in_vec((i-1)*3 + 3) = BiI(i,3)
      enddo

      call compute_hex8r_op_form_B(in_vec, out_vec)

      call unpack_2d_rowmajor(out_vec, 6, 24, B)
      end subroutine hex8r_op_form_B_wrapper

      subroutine hex8r_op_internal_force_wrapper(B, stress, detJ,
     1 weight, fint)
      implicit none
      real*8 B(6,24), stress(6), detJ, weight, fint(24)
      real*8 in_vec(152)

      call pack_2d_rowmajor(B, 6, 24, in_vec(1:144))
      in_vec(145:150) = stress
      in_vec(151) = detJ
      in_vec(152) = weight

      call compute_hex8r_op_internal_force(in_vec, fint)
      end subroutine hex8r_op_internal_force_wrapper

      subroutine mat_op_constitutive_linear_wrapper(E, nu, D)
      implicit none
      real*8 E, nu, D(6,6)
      real*8 in_vec(2), out_vec(36)

      in_vec(1) = E
      in_vec(2) = nu
      call compute_mat_op_constitutive_linear(in_vec, out_vec)

      call unpack_2d_rowmajor(out_vec, 6, 6, D)
      end subroutine mat_op_constitutive_linear_wrapper

      subroutine hex8r_op_kinematics_wrapper(F, J, B, B_bar, I1_bar_B,
     1 C, C_bar, Cinv, I1_bar_C, J_minus_2_3)
      implicit none
      real*8 F(3,3), J, B(3,3), B_bar(3,3), I1_bar_B
      real*8 C(3,3), C_bar(3,3), Cinv(3,3), I1_bar_C, J_minus_2_3
      real*8 in_vec(9), out_vec(49)

      call pack_2d_rowmajor(F, 3, 3, in_vec(1:9))

      call compute_hex8r_op_kinematics(in_vec, out_vec)

      J = out_vec(1)
      call unpack_2d_rowmajor(out_vec(2:10), 3, 3, B)
      call unpack_2d_rowmajor(out_vec(11:19), 3, 3, B_bar)
      I1_bar_B = out_vec(20)
      call unpack_2d_rowmajor(out_vec(21:29), 3, 3, C)
      call unpack_2d_rowmajor(out_vec(30:38), 3, 3, C_bar)
      call unpack_2d_rowmajor(out_vec(39:47), 3, 3, Cinv)
      I1_bar_C = out_vec(48)
      J_minus_2_3 = out_vec(49)
      end subroutine hex8r_op_kinematics_wrapper

      subroutine mat_op_stress_cauchy_n3_wrapper(J, B_bar, C10, C20,
     1 C30, D1, D2, D3, stress)
      implicit none
      real*8 J, B_bar(3,3), C10, C20, C30, D1, D2, D3, stress(6)
      real*8 in_vec(16), out_vec(27)

      in_vec(1) = J
      call pack_2d_rowmajor(B_bar, 3, 3, in_vec(2:10))
      in_vec(11) = C10
      in_vec(12) = C20
      in_vec(13) = C30
      in_vec(14) = D1
      in_vec(15) = D2
      in_vec(16) = D3

      call compute_mat_op_stress_cauchy_n3(in_vec, out_vec)

      stress = out_vec(1:6)
      end subroutine mat_op_stress_cauchy_n3_wrapper

      subroutine mat_op_stress_pk2_n3_wrapper(J, J_minus_2_3, Cinv,
     1 C_bar, I1_bar, C10, C20, C30, D1, D2, D3, stress)
      implicit none
      real*8 J, J_minus_2_3, Cinv(3,3), C_bar(3,3), I1_bar
      real*8 C10, C20, C30, D1, D2, D3, stress(6)
      real*8 in_vec(27), out_vec(36)

      in_vec(1) = J
      in_vec(2) = J_minus_2_3
      call pack_2d_rowmajor(Cinv, 3, 3, in_vec(3:11))
      call pack_2d_rowmajor(C_bar, 3, 3, in_vec(12:20))
      in_vec(21) = I1_bar
      in_vec(22) = C10
      in_vec(23) = C20
      in_vec(24) = C30
      in_vec(25) = D1
      in_vec(26) = D2
      in_vec(27) = D3

      call compute_mat_op_stress_pk2_n3(in_vec, out_vec)

      stress = out_vec(1:6)
      end subroutine mat_op_stress_pk2_n3_wrapper

      subroutine mat_op_dmat_n3_wrapper(B_bar, J, I1_bar, C10, C20,
     1 C30, D1, D2, D3, D)
      implicit none
      real*8 B_bar(3,3), J, I1_bar, C10, C20, C30, D1, D2, D3
      real*8 D(6,6)
      real*8 in_vec(17), out_vec(36)

      call pack_2d_rowmajor(B_bar, 3, 3, in_vec(1:9))
      in_vec(10) = J
      in_vec(11) = I1_bar
      in_vec(12) = C10
      in_vec(13) = C20
      in_vec(14) = C30
      in_vec(15) = D1
      in_vec(16) = D2
      in_vec(17) = D3

      call compute_mat_op_dmat_n3(in_vec, out_vec)

      call unpack_2d_rowmajor(out_vec, 6, 6, D)
      end subroutine mat_op_dmat_n3_wrapper

      subroutine mat_op_dmat_pk2_n3_wrapper(C_bar, Cinv, J, I1_bar,
     1 J_minus_2_3, C10, C20, C30, D1, D2, D3, D)
      implicit none
      real*8 C_bar(3,3), Cinv(3,3), J, I1_bar, J_minus_2_3
      real*8 C10, C20, C30, D1, D2, D3, D(6,6)
      real*8 in_vec(27), out_vec(36)

      call pack_2d_rowmajor(C_bar, 3, 3, in_vec(1:9))
      call pack_2d_rowmajor(Cinv, 3, 3, in_vec(10:18))
      in_vec(19) = J
      in_vec(20) = I1_bar
      in_vec(21) = J_minus_2_3
      in_vec(22) = C10
      in_vec(23) = C20
      in_vec(24) = C30
      in_vec(25) = D1
      in_vec(26) = D2
      in_vec(27) = D3

      call compute_mat_op_dmat_pk2_n3(in_vec, out_vec)

      call unpack_2d_rowmajor(out_vec, 6, 6, D)
      end subroutine mat_op_dmat_pk2_n3_wrapper

      subroutine hex8r_op_hourglass_gamma_wrapper(BiI, COORD, gammas)
      implicit none
      real*8 BiI(8,3), COORD(8,3), gammas(8,4)
      real*8 in_vec(48), out_vec(32)
      integer i

      do i = 1, 8
         in_vec((i-1)*3 + 1) = BiI(i,1)
         in_vec((i-1)*3 + 2) = BiI(i,2)
         in_vec((i-1)*3 + 3) = BiI(i,3)
         in_vec(24 + (i-1)*3 + 1) = COORD(i,1)
         in_vec(24 + (i-1)*3 + 2) = COORD(i,2)
         in_vec(24 + (i-1)*3 + 3) = COORD(i,3)
      enddo

      call compute_hex8r_op_hourglass_gamma(in_vec, out_vec)

      call unpack_2d_rowmajor(out_vec, 8, 4, gammas)
      end subroutine hex8r_op_hourglass_gamma_wrapper

      subroutine mat_op_rot_dmtx_wrapper(D, J0Inv, rj, D_rotated)
      implicit none
      real*8 D(6,6), J0Inv(3,3), rj, D_rotated(6,6)
      real*8 in_vec(46), out_vec(36)

      call pack_2d_rowmajor(D, 6, 6, in_vec(1:36))
      call pack_2d_rowmajor(J0Inv, 3, 3, in_vec(37:45))
      in_vec(46) = rj

      call compute_mat_op_rot_dmtx(in_vec, out_vec)

      call unpack_2d_rowmajor(out_vec, 6, 6, D_rotated)
      end subroutine mat_op_rot_dmtx_wrapper

      subroutine hex8r_op_k_matrices_wrapper(C_tilde, VOL, Kmat,
     1 K_alpha_u, K_alpha_alpha)
      implicit none
      real*8 C_tilde(6,6), VOL, Kmat(4,4,3,3), K_alpha_u(4,6,3)
      real*8 K_alpha_alpha(6,6)
      real*8 in_vec(37), out_vec(252)

      call pack_2d_rowmajor(C_tilde, 6, 6, in_vec(1:36))
      in_vec(37) = VOL

      call compute_hex8r_op_k_matrices(in_vec, out_vec)

      call unpack_4d_rowmajor(out_vec(1:144), 4, 4, 3, 3, Kmat)
      call unpack_3d_rowmajor(out_vec(145:216), 4, 6, 3, K_alpha_u)
      call unpack_2d_rowmajor(out_vec(217:252), 6, 6, K_alpha_alpha)
      end subroutine hex8r_op_k_matrices_wrapper

      end module hex8r_generated_wrappers
