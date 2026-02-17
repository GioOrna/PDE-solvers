#include "Stokes.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Stokes::dim;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  //TO_FILL
  // ∂u∂t + sigma u − mu ∆u + ∇p = f
  const auto sigma = [](const Point<dim> & /*p*/) { return 0.0; };
  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };
  const auto f  = [](const Point<dim> &, const double &){
    Tensor<1, dim> result;
    result[0]=0.0;
    result[1]=0.0;
    return result;
  };

  //TO_FILL
  Stokes problem(/*mesh_filename = */ "../mesh/mesh-square-h0.100000.msh",
               /* T = */ 1.0,
               /* delta_t = */ 0.1,
               /* deg vel */2,
               /* deg pre */1,
               mu,
               sigma,
               f);

  problem.run();

  return 0;
}