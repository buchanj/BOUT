
class LaplacePetsc;

#ifndef __PETSC_LAPLACE_H__
#define __PETSC_LAPLACE_H__

#ifndef BOUT_HAS_PETSC

#include <boutexception.hxx>
#include <invert_laplace.hxx>

class LaplacePetsc : public Laplacian {
public:
  LaplacePetsc(Options *opt = NULL) { throw BoutException("No PETSc solver available"); }
};

#else

#include <output.hxx>
#include <petscksp.h>
#include <options.hxx>
#include <invert_laplace.hxx>
#include <bout/petsclib.hxx>

class LaplacePetsc : public Laplacian {
public:
  LaplacePetsc(Options *opt = NULL);
  ~LaplacePetsc() {
    KSPDestroy( &ksp ); 
    VecDestroy( &xs );  
    VecDestroy( &bs ); 
    MatDestroy( &MatA );
  }
  
  void setCoefA(const Field2D &val) { A = val; }
  void setCoefC(const Field2D &val) { C = val; }
  void setCoefD(const Field2D &val) { D = val; }

  void setCoefA(const Field3D &val) { A = val; }
  void setCoefC(const Field3D &val) { C = val; }
  void setCoefD(const Field3D &val) { D = val; }
  
  const FieldPerp solve(const FieldPerp &b);
  const FieldPerp solve(const FieldPerp &b, const FieldPerp &x0);

  void Element(int i, int x, int z, int xshift, int zshift, PetscScalar ele, Mat &MatA );
  void Coeffs( int x, int y, int z, BoutReal &A1, BoutReal &A2, BoutReal &A3, BoutReal &A4, BoutReal &A5 );
  
private:
  Field3D A, C, D;

  FieldPerp sol; // solution
  
  // Istart is the first row of MatA owned by the process, Iend is 1 greater than the last row.
  int Istart, Iend; 

  int meshx, meshz, size, localN;
  MPI_Comm comm;
  Mat MatA;
  Vec xs, bs; // solution, RHS
  KSP ksp;

  PetscLib lib;
};

#endif //BOUT_HAS_PETSC_DEV

#endif //__PETSC_LAPLACE_H__
