#include <globals.hxx>
#include <boutexception.hxx>
#include "petsc_laplace.hxx"

#ifdef BOUT_HAS_PETSC

LaplacePetsc::LaplacePetsc(Options *opt) : Laplacian(opt) {

  // Get communicator for group of processors in X - all points in z-x plane for fixed y.
  comm = mesh->getXcomm();
  
  // Need to determine local size to use based on prior parallelisation
  // Coefficient values are stored only on local processors.
  
  localN = (mesh->xend - mesh->xstart + 1) * (mesh->ngz-1);
  if(mesh->firstX())
    localN += mesh->xstart * (mesh->ngz-1);    // If on first processor add on width of boundary region
  if(mesh->lastX())
    localN += mesh->xstart * (mesh->ngz-1);    // If on last processor add on width of boundary region
  
  // Calculate total number of points in physical grid
  if(MPI_Allreduce(&localN, &size, 1, MPI_INT, MPI_SUM, comm) != MPI_SUCCESS)
    throw BoutException("Error in MPI_Allreduce during LaplacePetsc initialisation");
  
  // Calculate total (physical) grid dimensions
  meshz = mesh->ngz-1;
  meshx = size / meshz;

  // Create Vectors 
  VecCreate( comm, &xs );                        
  VecSetSizes( xs, localN, size );                
  VecSetFromOptions( xs );                     
  VecDuplicate( xs , &bs );                   
  
  // Set size of Matrix on each processor to localN x localN
  MatCreate( comm, &MatA );                                
  MatSetSizes( MatA, localN, localN, size, size );                  
  MatSetFromOptions(MatA);                                       
  MatMPIAIJSetPreallocation( MatA,9, PETSC_NULL, 9, PETSC_NULL ); 
  MatSetUp(MatA); 

  // Declare KSP Context 
  KSPCreate( comm, &ksp ); 
}

const FieldPerp LaplacePetsc::solve(const FieldPerp &b) {
  return solve(b,b);
}

const FieldPerp LaplacePetsc::solve(const FieldPerp &b, const FieldPerp &x0) {

  int y = b.getIndex();           // Get the Y index
  sol = (FieldPerp) *b.clone();   // Initialize the solution field.

  // Set Matrix Elements

  // Determine which row/columns of the matrix are locally owned
  MatGetOwnershipRange( MatA, &Istart, &Iend );

  // Loop over locally owned rows of matrix A - i labels NODE POINT from bottom left (0,0) = 0 to top right (meshx-1,meshz-1) = meshx*meshz-1
  // i increments by 1 for an increase of 1 in Z and by meshz for an increase of 1 in X.
  int i = Istart;

  // X=0 to mesh->xstart-1 defines the boundary region of the domain.
  if( mesh->firstX() ) 
    {
      for(int x=0; x<mesh->xstart; x++)
	{
	  for(int z=0; z<mesh->ngz-1; z++) 
	    {
	      // Set Diagonal Values to 1
	      PetscScalar val = 1;
	      MatSetValues(MatA,1,&i,1,&i,&val,INSERT_VALUES);  
	      
	      // Set values corresponding to node adjacent in x to -1 if zero gradiaent condition is set.
	      val = -1;
	      if(flags & INVERT_AC_IN_GRAD) 
		Element(i,x,z, 1, 0, val, MatA ); 
	      
	      // Set Components of RHS
	      val=0;
	      if( flags & INVERT_IN_RHS )         val = b[x][z];
	      else if( flags & INVERT_IN_SET )    val = x0[x][z];
	      VecSetValues( bs, 1, &i, &val, INSERT_VALUES );
	      
	      i++; // Increment row in Petsc matrix
	    }
	}
    }

  // Main domain with Laplacian operator
  for(int x=mesh->xstart; x <= mesh->xend; x++)
    {
      for(int z=0; z<mesh->ngz-1; z++) 
	{
	  BoutReal A0, A1, A2, A3, A4, A5;
	  A0 = A[x][y][z];
	  Coeffs( x, y, z, A1, A2, A3, A4, A5 );

	  // Set Matrix Elements
	  // f(i,j) = f(x,z)
	  PetscScalar val = A0 - 2.0*( (A1 / pow(mesh->dx[x][y],2.0)) + (A2 / pow(mesh->dz,2.0)) ); 
	  MatSetValues(MatA,1,&i,1,&i,&val,INSERT_VALUES); 
      
	  // f(i-1,j-1)
	  val = A3 / 4.0*( mesh->dx[x][y] * mesh->dz ); 
	  Element(i,x,z, -1, -1, val, MatA ); 
      
	  // f(i,j-1)
	  val = A2/( pow( mesh->dz,2.0) ) - A5/( 2.0*mesh->dz ); 
	  Element(i,x,z, 0, -1, val, MatA ); 
      
	  // f(i+1,j-1)
	  val = -1.0*A3/( 4.0*mesh->dx[x][y]*mesh->dz); 
	  Element(i,x,z, 1, -1, val, MatA );
      
	  // f(i-1,j)
	  val = A1/( pow( mesh->dx[x][y],2.0) ) - A4/( 2.0*mesh->dx[x][y] ); 
	  Element(i,x,z, -1, 0, val, MatA );

	  // f(i+1,j)
	  val = A1/( pow( mesh->dx[x][y],2.0) ) + A4/( 2.0*mesh->dx[x][y] ); 
	  Element(i,x,z, 1, 0, val, MatA );
      
	  // f(i-1,j+1)
	  val = -1.0*A3/( 4*mesh->dx[x][y]*mesh->dz ); 
	  Element(i,x,z, -1, 1, val, MatA );
      
	  // f(i,j+1)
	  val = A2/( pow( mesh->dz,2.0) ) + A5/( 2.0*mesh->dz ); 
	  Element(i,x,z, 0, 1, val, MatA );

	  // f(i+1,j+1)
	  val = A3/( 4.0*mesh->dx[x][y]*mesh->dz ); 
	  Element(i,x,z, 1, 1, val, MatA );
      
	  // Set Components of RHS Vector
	  val  = b[x][z];
	  VecSetValues( bs, 1, &i, &val, INSERT_VALUES );
	  i++;
	}
    }

  // X=mesh->xend+1 to mesh->ngx-1 defines the upper boundary region of the domain.
  if( mesh->lastX() ) 
    {
      for(int x=mesh->xend+1; x<mesh->ngx; x++)
	{
	  for(int z=0; z<mesh->ngz-1; z++) 
	    {
	      PetscScalar val = 1; 
	      MatSetValues(MatA,1,&i,1,&i,&val,INSERT_VALUES);
	      
	      val = -1;
	      if(flags & INVERT_AC_OUT_GRAD) 
		Element(i,x,z, -1, 0, val, MatA );
	      
	      // Set Components of RHS
	      val=0;
	      if( flags & INVERT_OUT_RHS )        val = b[x][z];
	      else if( flags & INVERT_OUT_SET )   val = x0[x][z];
	      VecSetValues( bs, 1, &i, &val, INSERT_VALUES );
	      
	      i++; // Increment row in Petsc matrix
	    }
	}
    }

  if(i != Iend) {
    throw BoutException("Petsc index sanity check failed");
  }
 
  // Assemble Matrix
  MatAssemblyBegin( MatA, MAT_FINAL_ASSEMBLY );     
  MatAssemblyEnd( MatA, MAT_FINAL_ASSEMBLY );     

  // Assemble RHS Vector
  VecAssemblyBegin(bs);
  VecAssemblyEnd(bs);

  // Create Linear Solver
  KSPCreate( comm, &ksp );                   
  KSPSetOperators( ksp,MatA,MatA,DIFFERENT_NONZERO_PATTERN ); 
  KSPSetFromOptions( ksp );                      
 
  // Solve the system - FIXME add functionality to change solver type from options.
  KSPSolve( ksp, bs, xs );               

  // Add data to FieldPerp Object
  i = Istart;
  if(mesh->firstX()) 
    {
      for(int x=0; x<mesh->xstart; x++)
	{
	  for(int z=0; z<mesh->ngz-1; z++) 
	    {
	      PetscScalar val = 0;
	      VecGetValues(xs, 1, &i, &val ); 
	      sol[x][z] = val;
	      i++; // Increment row in Petsc matrix
	    }
	}
    }
  
  for(int x=mesh->xstart; x <= mesh->xend; x++)
    {
      for(int z=0; z<mesh->ngz-1; z++) 
	{
	  PetscScalar val = 0;
	  VecGetValues(xs, 1, &i, &val ); 
	  sol[x][z] = val;
	  i++; // Increment row in Petsc matrix
	}
    }

  if(mesh->lastX()) 
    {
      for(int x=mesh->xend+1; x<mesh->ngx; x++)
	{
	  for(int z=0;z < mesh->ngz-1; z++) 
	    {
	      PetscScalar val = 0;
	      VecGetValues(xs, 1, &i, &val ); 
	      sol[x][z] = val;
	      i++; // Increment row in Petsc matrix
	    }	
	}
    }
  
  if(i != Iend) {
    throw BoutException("Petsc index sanity check 2 failed");
  }

  return sol;
}

void LaplacePetsc::Element(int i, int x, int z, int xshift, int zshift, PetscScalar ele, Mat &MatA ) 
{
  // Need to convert LOCAL x to GLOBAL x in order to correctly calculate PETSC Matrix Index. D'oh!

  int xoffset = Istart / meshz;
  if( Istart % meshz != 0 )
    throw  BoutException("Petsc index sanity check 3 failed");

  int row_new = x + xshift; // should never be out of range.
  if( !mesh->firstX() ) row_new += (xoffset - mesh->xstart);

  int col_new = z + zshift;
  if( col_new < 0 )            col_new = meshz-1;
  else if( col_new > meshz-1 ) col_new = 0;
 
  // convert to global indices
  int index = (row_new * meshz) + col_new;
  
  MatSetValues(MatA,1,&i,1,&index,&ele,INSERT_VALUES); 
}

void LaplacePetsc::Coeffs( int x, int y, int z, BoutReal &coef1, BoutReal &coef2, BoutReal &coef3, BoutReal &coef4, BoutReal &coef5 )
{
  coef1 = mesh->g11[x][y];     // X 2nd derivative coefficient
  coef2 = mesh->g33[x][y];     // Z 2nd derivative coefficient
  coef3 = 2.*mesh->g13[x][y];  // X-Z mixed derivative coefficient
  
  coef4 = 0.0;
  coef5 = 0.0;
  if(all_terms) {
    coef4 = mesh->G1[x][y]; // X 1st derivative
    coef5 = mesh->G3[x][y]; // Z 1st derivative
  }
  
  coef1 *= D[x][y][z];
  coef2 *= D[x][y][z];
  coef3 *= D[x][y][z];
  coef4 *= D[x][y][z];
  coef5 *= D[x][y][z];
  
  if(nonuniform) 
    {
      // non-uniform mesh correction
      if((x != 0) && (x != (mesh->ngx-1))) 
	{
	  //coef4 += mesh->g11[jx][jy]*0.25*( (1.0/dx[jx+1][jy]) - (1.0/dx[jx-1][jy]) )/dx[jx][jy]; // SHOULD BE THIS (?)
	  //coef4 -= 0.5 * ( ( mesh->dx[x+1][y] - mesh->dx[x-1][y] ) / SQ ( mesh->dx[x][y] ) ) * coef1; // BOUT-06 term
	  coef4 -= 0.5 * ( ( mesh->dx[x+1][y] - mesh->dx[x-1][y] ) / pow( mesh->dx[x][y], 2.0 ) ) * coef1; // BOUT-06 term
	}
    }
  
  // A first order derivative term
  if((x > 0) && (x < (mesh->ngx-1)))
    coef4 += mesh->g11[x][y] * (C[x+1][y][z] - C[x-1][y][z]) / (2.*mesh->dx[x][y]*(C[x][y][z]));
  
  if(mesh->ShiftXderivs && mesh->IncIntShear) {
    // d2dz2 term
    coef2 += mesh->g11[x][y] * mesh->IntShiftTorsion[x][y] * mesh->IntShiftTorsion[x][y];
    // Mixed derivative
    coef3 = 0.0; // This cancels out
  }
  
  //  coef1 = coef1 / SQ(mesh->dx[x][y]);
  coef1 = coef1 / pow(mesh->dx[x][y], 2.0);
  coef3 = coef3 / 2.*mesh->dx[x][y];
  coef4 = coef4 / 2.*mesh->dx[x][y];
}


#endif // PETSC
