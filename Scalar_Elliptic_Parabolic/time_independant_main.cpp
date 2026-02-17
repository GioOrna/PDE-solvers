#include <iostream>

#include "Scalar_Elliptic_Parabolic.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  #ifdef USE_MPI
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  #endif

  constexpr unsigned int dim = DD::dim;

  const unsigned int N_el = 10; //don't use h, use N_el and calculate manually N_el
  const unsigned int r    = 1;
  //-(mu u')' + b u' + sigma u = f
  const auto         mu   = [](const Point<dim>           &/*p*/) { return 1.0; };
  const auto         b   = [](const Point<dim>           &p) { 
    Tensor<1,dim> res;
    res[0]=-p[0]; 
    return res;
  };
  const auto         sigma   = [](const Point<dim>           &/*p*/) { return 1.0; };
  const auto         f    = [](const Point<dim> &p) {
    return (2.+(p[0]*p[0]));
  };

  DD problem(N_el, r, mu, b, sigma, f);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output(0);

  return 0;
}