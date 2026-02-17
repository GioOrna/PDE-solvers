#include "Stokes.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Stokes::dim;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  //TO_FILL
  // sigma u − mu ∆u + ∇p = f
  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };
  const auto sigma = [](const Point<dim> & /*p*/) { return 0.0; };
  const auto f  = [](const Point<dim> & /*p*/){
    Tensor<1, dim> result;
    result[0]=0.0;
    result[1]=-100.0;
    return result;
  };
  //std::function<Tensor<1, dim>(const Point<dim> &, const double &)> f;


  Stokes problem(/*mesh_filename = */ "../mesh/mesh-pipe.msh",
               /* deg vel */2,
               /* deg pre */1,
               mu,
               sigma,
               f);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output(2);

  return 0;
}