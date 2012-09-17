
class LaplacePetsc;

#ifndef __PETSC_LAPLACE_H__
#define __PETSC_LAPLACE_H__

#ifndef BOUT_HAS_PETSC

#include <boutexception.hxx>
#include <invert_laplace.hxx>

class LaplacePetsc : public Laplacian {
public:
  LaplacePetsc(Options *opt = NULL) { throw BoutException("No PETSc solver available"); }

  void setCoefA(const Field2D &val) {}
  void setCoefC(const Field2D &val) {}
  void setCoefD(const Field2D &val) {}

  const FieldPerp solve(const FieldPerp &b) {}
};

#else

#include <globals.hxx>
#include <output.hxx>
#include <petscksp.h>
#include <options.hxx>
#include <invert_laplace.hxx>
#include <bout/petsclib.hxx>
#include <boutexception.hxx>

class LaplacePetsc : public Laplacian {
public:
  LaplacePetsc(Options *opt = NULL);
  ~LaplacePetsc() {
    KSPDestroy( &ksp ); 
    VecDestroy( &xs );  
    VecDestroy( &bs ); 
    MatDestroy( &MatA );
  }
  
  void setCoefA(const Field2D &val) { A = val; coefchanged = true;}
  void setCoefC(const Field2D &val) { C = val; coefchanged = true;}
  void setCoefD(const Field2D &val) { D = val; coefchanged = true;}

  void setCoefA(const Field3D &val) { A = val; coefchanged = true;}
  void setCoefC(const Field3D &val) { C = val; coefchanged = true;}
  void setCoefD(const Field3D &val) { D = val; coefchanged = true;}
  
  const FieldPerp solve(const FieldPerp &b);
  const FieldPerp solve(const FieldPerp &b, const FieldPerp &x0);

  void Element(int i, int x, int z, int xshift, int zshift, PetscScalar ele, Mat &MatA );
  void Coeffs( int x, int y, int z, BoutReal &A1, BoutReal &A2, BoutReal &A3, BoutReal &A4, BoutReal &A5 );
  
private:
  Field3D A, C, D;
  bool coefchanged;           // Set to true when A,C or D coefficients are changed
  int lastflag;               // The flag used to construct the matrix

  FieldPerp sol;              // solution Field
  
  // Istart is the first row of MatA owned by the process, Iend is 1 greater than the last row.
  int Istart, Iend; 

  int meshx, meshz, size, localN;
  MPI_Comm comm;
  Mat MatA;
  Vec xs, bs;                 // Solution and RHS vectors
  KSP ksp;
  
  Options *opts;              // Laplace Section Options Object
  KSPType ksptype;            // Solver Type;

  // Values specific to particular solvers
  BoutReal richardson_damping_factor;  
  BoutReal chebyshev_max, chebyshev_min;
  int gmres_max_steps;

  // Convergence Parameters. Solution is considered converged if |r_k| < max( rtol * |b| , atol )
  // where r_k = b - Ax_k. The solution is considered diverged if |r_k| > dtol * |b|.
  BoutReal rtol, atol, dtol;
  int maxits; // Maximum number of iterations in solver.

  PetscLib lib;
};

#endif //BOUT_HAS_PETSC_DEV

#endif //__PETSC_LAPLACE_H__
