#include <iostream>
#include "Scalar_Elliptic_Parabolic.hpp"

constexpr unsigned int dim = DD::dim;

// Main function.
int
main(int argc, char * argv[])
{
  #ifdef USE_MPI
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  #endif
  const unsigned int N_el = 39; //don't use h, use N_el and calculate manually N_el
  const unsigned int r    = 1;
  //-(mu u')' + b u' + sigma u = f
  const auto         mu   = [](const Point<dim>           &p) {return 0.1;};
  const auto         b   = [](const Point<dim>           &p) { 
    Tensor<1,dim> res;
    for(int i=0;i<dim;i++){
    res[i]=0.0; 
    }
    return res;
  };
  const auto         sigma   = [](const Point<dim>           &/*p*/) { return 0.0; };
  const auto         f    = [](const Point<dim> &p, const double & /*t*/) {
    return 0.0;
  };

  DD problem("../mesh/mesh-cube-10.msh",
    /* T = */ 1.0,
    /* theta = */ 0.0,
    /* delta_t */ 0.0025,
                r, mu, b, sigma, f);

  problem.run();

  return 0;
}