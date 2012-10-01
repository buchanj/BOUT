
#include "rk4.hxx"

#include <boutcomm.hxx>
#include <utils.hxx>
#include <boutexception.hxx>
#include <msg_stack.hxx>

#include <cmath>

#include <output.hxx>

RK4Solver::RK4Solver() : Solver() {
  
}

RK4Solver::~RK4Solver() {

}

void RK4Solver::setMaxTimestep(BoutReal dt) {
  if(dt > timestep)
    return; // Already less than this
  
  timestep = dt; // Won't be used this time, but next
}

int RK4Solver::init(rhsfunc f, bool restarting, int nout, BoutReal tstep) {

  int msg_point = msg_stack.push("Initialising RK4 solver");
  
  /// Call the generic initialisation first
  if(Solver::init(f, restarting, nout, tstep))
    return 1;
  
  output << "\n\tRunge-Kutta 4th-order solver\n";

  nsteps = nout; // Save number of output steps
  out_timestep = tstep;

  // Choose timestep
  if(max_dt < 0.0) {
    max_dt = tstep;
    output << "\tWARNING: Starting dt not set. Starting with dt = " << max_dt << endl;
  }
  
  // Calculate number of variables
  nlocal = getLocalN();
  
  // Get total problem size
  if(MPI_Allreduce(&nlocal, &neq, 1, MPI_INT, MPI_SUM, BoutComm::get())) {
    throw BoutException("MPI_Allreduce failed!");
  }
  
  output.write("\t3d fields = %d, 2d fields = %d neq=%d, local_N=%d\n",
	       n3Dvars(), n2Dvars(), neq, nlocal);
  
  // Allocate memory
  f0 = new BoutReal[nlocal];
  f1 = new BoutReal[nlocal];
  f2 = new BoutReal[nlocal];
  
  // Put starting values into f0
  save_vars(f0);
  
  // Get options
  Options *options = Options::getRoot();
  options = options->getSection("solver");
  OPTION(options, atol, 1.e-5); // Absolute tolerance
  OPTION(options, rtol, 1.e-3); // Relative tolerance
  OPTION(options, max_timestep, tstep); // Maximum timestep
  OPTION(options, start_timestep, -1.); // Starting timestep
  OPTION(options, mxstep, 500); // Maximum number of steps between outputs
  
  msg_stack.pop(msg_point);

  return 0;
}

int RK4Solver::run(MonitorFunc monitor) {
  int msg_point = msg_stack.push("RK4Solver::run()");
  
  timestep = out_timestep;
  if((max_timestep > 0.) && (timestep > max_timestep))
    timestep = max_timestep;
  if(start_timestep > 0.) 
    timestep = start_timestep;
  
  for(int s=0;s<nsteps;s++) {
    BoutReal target = simtime + out_timestep;
    
    BoutReal dt;
    bool running = true;
    int internal_steps = 0;
    do {
      // Take a step within the error
      do {
        dt = timestep;
        if((simtime + dt) >= target) {
          dt = target - simtime; // Make sure the last timestep is on the output 
          running = false;
        }
        // Take two half-steps
        take_step(simtime,          0.5*dt, f0, f1);
        take_step(simtime + 0.5*dt, 0.5*dt, f1, f2);
        
        // Take a full step
        take_step(simtime, dt, f0, f1);
        
        // Check accuracy
        BoutReal local_err = 0.;
        #pragma omp parallel for reduction(+: local_err)   
        for(int i=0;i<nlocal;i++)
          local_err += fabs(f2[i] - f1[i]) / ( fabs(f1[i] + f2[i]) + atol );
        
        // Average over all processors
        BoutReal err;
        if(MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_SUM, BoutComm::get())) {
          throw BoutException("MPI_Allreduce failed");
        }

        err /= (BoutReal) neq;
        
        internal_steps++;
        if(internal_steps > mxstep)
          throw BoutException("ERROR: MXSTEP exceeded. timestep = %e, err=%e\n", timestep, err);

        //output.write("ERROR: t=%e dt=%e err=%e, atol=%e\n", simtime, dt, err, rtol);
        if(!running && (err < 0.1*rtol)) {
          break;
        }else if((err > rtol) || (err < 0.1*rtol)) {
          // Need to change timestep. Error ~ dt^5
          timestep /= pow(err / (0.5*rtol), 0.2);
          running = true; // Keep running
          
          if((max_timestep > 0) && (timestep > max_timestep))
            timestep = max_timestep;
        }else {
          break; // Acceptable accuracy
        }
      }while(true);
      
      // Taken a step, swap buffers
      swap(f2, f0);
      simtime += dt;
    }while(running);
    
    iteration++; // Advance iteration number
    
    /// Write the restart file
    restart.write("%s/BOUT.restart.%s", restartdir.c_str(), restartext.c_str());
    
    if((archive_restart > 0) && (iteration % archive_restart == 0)) {
      restart.write("%s/BOUT.restart_%04d.%s", restartdir.c_str(), iteration, restartext.c_str());
    }
    
    /// Call the monitor function
    
    if(monitor(this, simtime, s, nsteps)) {
      // User signalled to quit
      
      // Write restart to a different file
      restart.write("%s/BOUT.final.%s", restartdir.c_str(), restartext.c_str());
      
      output.write("Monitor signalled to quit. Returning\n");
      break;
    }
    
    // Reset iteration and wall-time count
    rhs_ncalls = 0;
  }
  
  msg_stack.pop(msg_point);
  
  return 0;
}

void RK4Solver::take_step(BoutReal curtime, BoutReal dt, BoutReal *start, BoutReal *result) { 
  static int n = 0;
  static BoutReal *k1, *k2, *k3, *k4, *tmp;
  
  if(n < nlocal) {
    k1 = new BoutReal[nlocal];
    k2 = new BoutReal[nlocal];
    k3 = new BoutReal[nlocal];
    k4 = new BoutReal[nlocal];
    tmp = new BoutReal[nlocal];
    n = nlocal;
  }
  
  load_vars(start);
  run_rhs(curtime);
  save_derivs(k1);
  
  #pragma omp parallel for
  for(int i=0;i<nlocal;i++)
    tmp[i] = f0[i] + 0.5*dt*k1[i];
  
  load_vars(tmp);
  run_rhs(curtime + 0.5*dt);
  save_derivs(k2);
  
  #pragma omp parallel for 
  for(int i=0;i<nlocal;i++)
    tmp[i] = f0[i] + 0.5*dt*k2[i];
  
  load_vars(tmp);
  run_rhs(curtime + 0.5*dt);
  save_derivs(k3);
 
  #pragma omp parallel for
  for(int i=0;i<nlocal;i++)
    tmp[i] = f0[i] + dt*k3[i];
  
  load_vars(tmp);
  run_rhs(curtime + dt);
  save_derivs(k4);
  
  #pragma omp parallel for
  for(int i=0;i<nlocal;i++)
    result[i] = start[i] + (1./6.)*dt*(k1[i] + 2.*k2[i] + 2.*k3[i] + k4[i]);
}
