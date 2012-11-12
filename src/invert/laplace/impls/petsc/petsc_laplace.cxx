#include "petsc_laplace.hxx"

#define KSP_RICHARDSON  "richardson"
#define KSP_CHEBYSHEV   "chebyshev"
#define KSP_CG          "cg"
#define KSP_GMRES       "gmres"
#define KSP_TCQMR       "tcqmr"
#define KSP_BCGS        "bcgs"
#define KSP_CGS         "cgs"
#define KSP_TFQMR       "tfqmr"
#define KSP_CR          "cr"
#define KSP_LSQR        "lsqr"
#define KSP_BICG        "bicg"
#define KSP_PREONLY     "preonly"

#ifdef BOUT_HAS_PETSC

LaplacePetsc::LaplacePetsc(Options *opt) : Laplacian(opt) {

  // Get Options in Laplace Section
  opts = Options::getRoot()->getSection("laplace");

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
  MatMPIAIJSetPreallocation( MatA,25, PETSC_NULL, 25, PETSC_NULL );
  MatSetUp(MatA); 

  // Declare KSP Context 
  KSPCreate( comm, &ksp ); 

  // Get KSP Solver Type
  string type;
  opts->get("ksptype", type, KSP_GMRES);
  
  if(strcasecmp(type.c_str(), KSP_RICHARDSON) == 0)     ksptype = KSPRICHARDSON;
  else if(strcasecmp(type.c_str(), KSP_CHEBYSHEV) == 0) ksptype = KSPCHEBYSHEV;
  else if(strcasecmp(type.c_str(), KSP_CG) == 0)        ksptype = KSPCG;
  else if(strcasecmp(type.c_str(), KSP_GMRES) == 0)     ksptype = KSPGMRES;
  else if(strcasecmp(type.c_str(), KSP_TCQMR) == 0)     ksptype = KSPTCQMR;
  else if(strcasecmp(type.c_str(), KSP_BCGS) == 0)      ksptype = KSPBCGS;
  else if(strcasecmp(type.c_str(), KSP_CGS) == 0)       ksptype = KSPCGS;
  else if(strcasecmp(type.c_str(), KSP_TFQMR) == 0)     ksptype = KSPTFQMR;
  else if(strcasecmp(type.c_str(), KSP_CR) == 0)        ksptype = KSPCR;
  else if(strcasecmp(type.c_str(), KSP_LSQR) == 0)      ksptype = KSPLSQR;
  else if(strcasecmp(type.c_str(), KSP_BICG) == 0)      ksptype = KSPBICG;
  else if(strcasecmp(type.c_str(), KSP_PREONLY) == 0)   ksptype = KSPPREONLY;
  else 
    throw BoutException("Unknown Krylov solver type '%s'", type.c_str());

  // Get Options specific to particular solver types
  opts->get("richardson_damping_factor",richardson_damping_factor,1.0,true);
  opts->get("chebyshev_max",chebyshev_max,100,true);
  opts->get("chebyshev_min",chebyshev_min,0.01,true);
  opts->get("gmres_max_steps",gmres_max_steps,30,true);
 
  // Get Tolerances for KSP solver
  opts->get("rtol",rtol,pow(10.0,-5),true);
  opts->get("atol",atol,pow(10.0,-50),true);
  opts->get("dtol",dtol,pow(10.0,5),true);
  opts->get("maxits",maxits,pow(10.0,5),true);

  // Ensure that the matrix is constructed first time
  coefchanged = true;
  lastflag = -1;
}

const FieldPerp LaplacePetsc::solve(const FieldPerp &b) {
  return solve(b,b);
}

const FieldPerp LaplacePetsc::solve(const FieldPerp &b, const FieldPerp &x0) {

  int y = b.getIndex();           // Get the Y index
  sol = (FieldPerp) *b.clone();   // Initialize the solution field.

  // Determine which row/columns of the matrix are locally owned
  MatGetOwnershipRange( MatA, &Istart, &Iend );

  int i = Istart;
  
  if(coefchanged || (flags != lastflag)) { // Coefficients or settings changed
    // Set Matrix Elements

    // Loop over locally owned rows of matrix A - i labels NODE POINT from bottom left (0,0) = 0 to top right (meshx-1,meshz-1) = meshx*meshz-1
    // i increments by 1 for an increase of 1 in Z and by meshz for an increase of 1 in X.
    
    // X=0 to mesh->xstart-1 defines the boundary region of the domain.
    if( mesh->firstX() ) 
      {
        for(int x=0; x<mesh->xstart; x++)
          {
            for(int z=0; z<mesh->ngz-1; z++) 
              {
                // Set Diagonal Values to 1
                PetscScalar val = 1;
		Element(i,x,z, 0, 0, val, MatA );
                
                // Set values corresponding to nodes adjacent in x if Neumann Boundary Conditions are required.
                if(flags & INVERT_AC_IN_GRAD) 
		  {
		    if( flags & INVERT_4TH_ORDER )
		      {
			// Fourth Order Accuracy on Boundary
			Element(i,x,z, 0, 0, -25.0 / (12.0*mesh->dx[x][y]), MatA ); 
			Element(i,x,z, 1, 0,   4.0 / mesh->dx[x][y], MatA ); 
			Element(i,x,z, 2, 0,  -3.0 / mesh->dx[x][y], MatA );
			Element(i,x,z, 3, 0,   4.0 / (3.0*mesh->dx[x][y]), MatA ); 
			Element(i,x,z, 4, 0,  -1.0 / (4.0*mesh->dx[x][y]), MatA );
		      }
		    else
		      {
			// Second Order Accuracy on Boundary
			Element(i,x,z, 0, 0, -3.0 / (2.0*mesh->dx[x][y]), MatA ); 
			Element(i,x,z, 1, 0,  2.0 / mesh->dx[x][y], MatA ); 
			Element(i,x,z, 2, 0, -1.0 / (2.0*mesh->dx[x][y]), MatA ); 
			Element(i,x,z, 3, 0, 0.0, MatA );  // Reset these elements to 0 in case 4th order flag was used previously
			Element(i,x,z, 4, 0, 0.0, MatA );
		      }
		  }
		else
		  {
		    // Set off diagonal elements to zero
		    Element(i,x,z, 1, 0, 0.0, MatA );
		    Element(i,x,z, 2, 0, 0.0, MatA );
		    Element(i,x,z, 3, 0, 0.0, MatA );
		    Element(i,x,z, 4, 0, 0.0, MatA );
		  }
                
                // Set Components of RHS and trial solution
                val=0;
                if( flags & INVERT_IN_RHS )         val = b[x][z];
                else if( flags & INVERT_IN_SET )    val = x0[x][z];
                VecSetValues( bs, 1, &i, &val, INSERT_VALUES );

		val = x0[x][z];
		VecSetValues( xs, 1, &i, &val, INSERT_VALUES );
                
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
	    
	    BoutReal dx   = mesh->dx[x][y];
	    BoutReal dx2  = pow( mesh->dx[x][y] , 2.0 );
	    BoutReal dz   = mesh->dz;
	    BoutReal dz2  = pow( mesh->dz, 2.0 );
	    BoutReal dxdz = mesh->dx[x][y] * mesh->dz;
            
            // Set Matrix Elements
            // f(i,j) = f(x,z)
            PetscScalar val = A0 - 2.0*( (A1 / dx2) + (A2 / dz2) ); 
	    if( flags & INVERT_4TH_ORDER ) val = A0 - (5.0/2.0)*( (A1 / dx2) + (A2 / dz2) );
	    Element(i,x,z, 0, 0, val, MatA );
            
	    // f(i-2,j-2)
	    val = 0;
	    if( flags & INVERT_4TH_ORDER ) val = A3 / ( 144.0 * dxdz );
	    Element(i,x,z, -2, -2, val, MatA ); 

	    // f(i-2,j-1)
	    if( flags & INVERT_4TH_ORDER ) val = -1.0 * A3 / ( 18.0 * dxdz );
	    Element(i,x,z, -2, -1, val, MatA );

	    // f(i-2,j)
	    if( flags & INVERT_4TH_ORDER ) val = (1.0/12.0) * ( (-1.0 * A1 /  dx2 ) + (A4 / dx) ); 
	    Element(i,x,z, -2, 0, val, MatA );

	    // f(i-2,j+1)
	    if( flags & INVERT_4TH_ORDER ) val = A3 / ( 18.0 * dxdz );
	    Element(i,x,z, -2, 1, val, MatA );

	    // f(i-2,j+2)
	    if( flags & INVERT_4TH_ORDER ) val = -1.0 * A3 / ( 144.0 * dxdz );
	    Element(i,x,z, -2, 2, val, MatA ); 

	    // f(i-1,j-2)
	    if( flags & INVERT_4TH_ORDER ) val = -1.0 * A3 / ( 18.0 * dxdz );
	    Element(i,x,z, -1, -2, val, MatA );

            // f(i-1,j-1)
            val = A3 / (4.0 * dxdz); 
	    if( flags & INVERT_4TH_ORDER ) val = 4.0 * A3 / ( 9.0 * dxdz ); 
            Element(i,x,z, -1, -1, val, MatA ); 

            // f(i-1,j)
            val = ( A1 / dx2 ) - A4 / ( 2.0 * dx ); 
	    if( flags & INVERT_4TH_ORDER ) val = ( 4.0 * A1 / ( 3.0 * dx2 ) ) - ( 2.0 * A4 / ( 3.0 * dx ) );
            Element(i,x,z, -1, 0, val, MatA );

	    // f(i-1,j+1)
            val = -1.0 * A3 / ( 4.0 * dxdz ); 
	    if( flags & INVERT_4TH_ORDER ) val = -4.0 * A3 / ( 9.0 * dxdz ); 
            Element(i,x,z, -1, 1, val, MatA );

	    // f(i-1,j+2)
	    val = 0;
	    if( flags & INVERT_4TH_ORDER ) val = A3 / ( 18.0 * dxdz );
	    Element(i,x,z, -1, 2, val, MatA );

	    // f(i,j-2)
	    if( flags & INVERT_4TH_ORDER ) val = (1.0/12.0) * ( ( -1.0 * A2 / dz2 ) + ( A5 / dz ) ); 
	    Element(i,x,z, 0, -2, val, MatA );
            
            // f(i,j-1)
            val = ( A2 / dz2 ) - ( A5 / ( 2.0 * dz ) ); 
	    if( flags & INVERT_4TH_ORDER ) val = ( 4.0 * A2 / ( 3.0 * dz2 ) ) - ( 2.0 * A5 / ( 3.0 * dz ) );
            Element(i,x,z, 0, -1, val, MatA ); 

            // f(i,j+1)
            val = ( A2 / dz2 ) + ( A5 / ( 2.0 * dz ) ); 
	    if( flags & INVERT_4TH_ORDER ) val = ( 4.0 * A2 / ( 3.0 * dz2 ) ) + ( 2.0 * A5 / ( 3.0 * dz ) );
            Element(i,x,z, 0, 1, val, MatA );

	    // f(i,j+2)
	    val = 0;
	    if( flags & INVERT_4TH_ORDER ) val = (-1.0/12.0) * ( ( A2 / dz2 ) + ( A5 / dz ) ); 
	    Element(i,x,z, 0, 2, val, MatA );

	    // f(i+1,j-2)
	    if( flags & INVERT_4TH_ORDER ) val = A3 / ( 18.0 * dxdz );
	    Element(i,x,z, 1, -2, val, MatA );
            
            // f(i+1,j-1)
            val = -1.0 * A3 / ( 4.0 * dxdz ); 
	    if( flags & INVERT_4TH_ORDER ) val = -4.0 * A3 / ( 9.0 * dxdz );
            Element(i,x,z, 1, -1, val, MatA );
            
            // f(i+1,j)
            val = ( A1 / dx2 ) + ( A4 / ( 2.0 * dx ) ); 
	    if( flags & INVERT_4TH_ORDER ) val = ( 4.0 * A1 / ( 3.0*dx2 ) ) + ( 2.0 * A4 / ( 3.0 * dx ) ); 
            Element(i,x,z, 1, 0, val, MatA );
            
            // f(i+1,j+1)
            val = A3 / ( 4.0 * dxdz ); 
	    if( flags & INVERT_4TH_ORDER ) val = 4.0 * A3 / ( 9.0 * dxdz ); 
            Element(i,x,z, 1, 1, val, MatA );

	    // f(i+1,j+2)
	    val = 0;
	    if( flags & INVERT_4TH_ORDER ) val = -1.0 * A3 / ( 18.0 * dxdz );
	    Element(i,x,z, 1, 2, val, MatA );

	    // f(i+2,j-2)
	    if( flags & INVERT_4TH_ORDER ) val = -1.0 * A3 / ( 144.0 * dxdz );
	    Element(i,x,z, 2, -2, val, MatA ); 

	    // f(i+2,j-1)
	    if( flags & INVERT_4TH_ORDER ) val = A3 / ( 18.0 * dxdz );
	    Element(i,x,z, 2, -1, val, MatA );

	    // f(i+2,j)
	    if( flags & INVERT_4TH_ORDER ) val = (-1.0/12.0) * ( (A1 / dx2) + (A4 / dx) ); 
	    Element(i,x,z, 2, 0, val, MatA );

	    // f(i+2,j+1)
	    if( flags & INVERT_4TH_ORDER ) val = -1.0 * A3 / ( 18.0 * dxdz );
	    Element(i,x,z, 2, 1, val, MatA );

	    // f(i+2,j+2)
	    if( flags & INVERT_4TH_ORDER ) val = A3 / ( 144.0 * dxdz );
	    Element(i,x,z, 2, 2, val, MatA ); 
            
            // Set Components of RHS Vector
            val  = b[x][z];
            VecSetValues( bs, 1, &i, &val, INSERT_VALUES );

	    // Set Components of Trial Solution Vector
	    val = x0[x][z];
	    VecSetValues( xs, 1, &i, &val, INSERT_VALUES ); 
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
                // Set Diagonal Values to 1
                PetscScalar val = 1;
		Element(i,x,z, 0, 0, val, MatA ); 
                
                // Set values corresponding to nodes adjacent in x if Neumann Boundary Conditions are required.
		if(flags & INVERT_AC_IN_GRAD) 
		  {
		    if( flags & INVERT_4TH_ORDER )
		      {
			// Fourth Order Accuracy on Boundary
			Element(i,x,z,  0, 0, 25.0 / (12.0*mesh->dx[x][y]), MatA ); 
			Element(i,x,z, -1, 0, -4.0 / mesh->dx[x][y], MatA ); 
			Element(i,x,z, -2, 0,  3.0 / mesh->dx[x][y], MatA );
			Element(i,x,z, -3, 0, -4.0 / (3.0*mesh->dx[x][y]), MatA ); 
			Element(i,x,z, -4, 0,  1.0 / (4.0*mesh->dx[x][y]), MatA );
		      }
		    else
		      {
			// Second Order Accuracy on Boundary
			Element(i,x,z,  0, 0,  3.0 / (2.0*mesh->dx[x][y]), MatA ); 
			Element(i,x,z, -1, 0, -2.0 / mesh->dx[x][y], MatA ); 
			Element(i,x,z, -2, 0,  1.0 / (2.0*mesh->dx[x][y]), MatA ); 
			Element(i,x,z, -3, 0,  0.0, MatA );  // Reset these elements to 0 in case 4th order flag was used previously
			Element(i,x,z, -4, 0,  0.0, MatA );
		      }
		  }
		else
		  {
		    // Set off diagonal elements to zero
		    Element(i,x,z, -1, 0, 0.0, MatA );
		    Element(i,x,z, -2, 0, 0.0, MatA );
		    Element(i,x,z, -3, 0, 0.0, MatA );
		    Element(i,x,z, -4, 0, 0.0, MatA );
		  }            
                
                // Set Components of RHS
                val=0;
                if( flags & INVERT_OUT_RHS )        val = b[x][z];
                else if( flags & INVERT_OUT_SET )   val = x0[x][z];
                VecSetValues( bs, 1, &i, &val, INSERT_VALUES );

		val = x0[x][z];
		VecSetValues( xs, 1, &i, &val, INSERT_VALUES );
                
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

    // Record which flags were used for this matrix
    lastflag = flags;
  }else {
    // Matrix hasn't changed. Only need to set Vec values
    
    if( mesh->firstX() ) 
      {
        for(int x=0; x<mesh->xstart; x++)
          {
            for(int z=0; z<mesh->ngz-1; z++) 
              {
                // Set Components of RHS and Trial Solution
                PetscScalar  val=0;
                if( flags & INVERT_IN_RHS )         val = b[x][z];
                else if( flags & INVERT_IN_SET )    val = x0[x][z];
                VecSetValues( bs, 1, &i, &val, INSERT_VALUES );

		val = x0[x][z];
		VecSetValues( xs, 1, &i, &val, INSERT_VALUES );
                
                i++; // Increment row in Petsc matrix
              }
          }
      }

    for(int x=mesh->xstart; x <= mesh->xend; x++)
      {
        for(int z=0; z<mesh->ngz-1; z++) 
          {
            // Set Components of RHS Vector and Trial Solution
            PetscScalar val  = b[x][z];
            VecSetValues( bs, 1, &i, &val, INSERT_VALUES );

	    val = x0[x][z];
	    VecSetValues( xs, 1, &i, &val, INSERT_VALUES );

            i++;
          }
      }

    if( mesh->lastX() ) 
      {
        for(int x=mesh->xend+1; x<mesh->ngx; x++)
          {
            for(int z=0; z<mesh->ngz-1; z++) 
              {
                PetscScalar val=0;
                if( flags & INVERT_OUT_RHS )        val = b[x][z];
                else if( flags & INVERT_OUT_SET )   val = x0[x][z];
                VecSetValues( bs, 1, &i, &val, INSERT_VALUES );

		val = x0[x][z];
		VecSetValues( xs, 1, &i, &val, INSERT_VALUES );
                
                i++; // Increment row in Petsc matrix
              }
          }
      }
    if(i != Iend) {
      throw BoutException("Petsc index sanity check failed");
    }
  }

  // Assemble RHS Vector
  VecAssemblyBegin(bs);
  VecAssemblyEnd(bs);

  // Assemble Trial Solution Vector
  VecAssemblyBegin(xs);
  VecAssemblyEnd(xs);

  // Configure Linear Solver               
  KSPSetOperators( ksp,MatA,MatA,DIFFERENT_NONZERO_PATTERN ); 
  KSPSetType( ksp, ksptype );

  if( ksptype == KSPRICHARDSON )     KSPRichardsonSetScale( ksp, richardson_damping_factor );
  else if( ksptype == KSPCHEBYSHEV ) KSPChebyshevSetEigenvalues( ksp, chebyshev_max, chebyshev_min );
  else if( ksptype == KSPGMRES )     KSPGMRESSetRestart( ksp, gmres_max_steps );

  KSPSetTolerances( ksp, rtol, atol, dtol, maxits );
  if( !( flags & INVERT_START_NEW ) ) KSPSetInitialGuessNonzero( ksp, (PetscBool) true );
  KSPSetFromOptions( ksp );                
 
  // Solve the system
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
  if( col_new < 0 )            col_new += meshz;
  else if( col_new > meshz-1 ) col_new -= meshz;
 
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
  if( (x > 0) && (x < (mesh->ngx-1)) )
    {
      if( C[x][y][z] != 0 )
	{
	  coef4 += mesh->g11[x][y] * (C[x+1][y][z] - C[x-1][y][z]) / (2.*mesh->dx[x][y]*(C[x][y][z]));
	  coef5 += mesh->g13[x][y] * (C[x+1][y][z] - C[x-1][y][z]) / (2.*mesh->dx[x][y]*(C[x][y][z]));
	}
    }
  
  if(mesh->ShiftXderivs && mesh->IncIntShear) {
    // d2dz2 term
    coef2 += mesh->g11[x][y] * mesh->IntShiftTorsion[x][y] * mesh->IntShiftTorsion[x][y];
    // Mixed derivative
    coef3 = 0.0; // This cancels out
  }
}


#endif // PETSC
