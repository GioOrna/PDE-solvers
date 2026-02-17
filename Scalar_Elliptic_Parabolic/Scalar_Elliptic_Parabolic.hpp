//TO_FILL
//#define TIME_DEPENDANT
#define DIRICHLET
//#define EXTERNAL_MESH
//#define USE_MPI
//#define NEUMANN
//#define D_D
#ifdef USE_MPI
  #ifdef D_D
    #error "USE_MPI cannot be defined if D_D is defined"
  #endif
#endif
#ifdef TIME_DEPENDANT
 #ifndef USE_MPI
    #error "use MPI when TIME_DEPENDANT"
 #endif
#endif

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
//TO_FILL
class DD
{
public:
  //TO_FILL
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 1;

  //TO_FILL
  #ifdef TIME_DEPENDANT
  // Initial condition.
  class FunctionU0 : public Function<dim>
  {
  public:
    // Constructor.
    FunctionU0() = default;
    // Evaluation of the function.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return p[0] * (1.0 - p[0]) * //
             p[1] * (1.0 - p[1]) * //
             p[2] * (1.0 - p[2]);
    }
  };
  #endif

  // MY_OPTION
  // IF WE HAVE CUSTOM Dirichlet boundary function.!!!
  #ifdef DIRICHLET
  //TO_FILL: leave empty if no CUSTOM boundary
  // This is implemented as a dealii::Function<dim>, instead of e.g. a lambda
  // function, because this allows to use dealii boundary utilities directly.
  class FunctionG : public Function<dim>
  {
  public:
    // Constructor.
    FunctionG()
    {}
    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };
  #endif

  // Constructor.
  DD(
    #ifdef D_D
            const unsigned int &subdomain_id_,
    #endif
    #ifndef EXTERNAL_MESH
    //MY_OPTION_1, if i generate the mesh
            const unsigned int      
                                    &N_el_,
    #else
            //MY_OPTION_2, if i take an input mesh
            const std::string  &mesh_file_name_,
    #endif

            //MY_OPTION, if time dependant
    #ifdef TIME_DEPENDANT
            const double  &T_,
            const double &theta_,
            const double &delta_t_,
    #endif

            const unsigned int &r_,
            const std::function<double(const Point<dim> &)> &mu_,
            const std::function<Tensor<1,dim>(const Point<dim> &)> &b_,
            const std::function<double(const Point<dim> &)> &sigma_,
            //MY_OPTION1 if time independant
            #ifndef TIME_DEPENDANT
            const std::function<double(const Point<dim> &)> &f_
            #else
            const std::function<double(const Point<dim> &, const double &)> &f_
            #endif
  ): 
    #ifndef EXTERNAL_MESH
    N_el(N_el_)
    #else
    mesh_file_name(mesh_file_name_)
    #endif
    #ifdef D_D
    , subdomain_id(subdomain_id_)
    #endif
    #ifdef TIME_DEPENDANT
    , T(T_)
    , theta(theta_)
    , delta_t(delta_t_)
    #endif
    , r(r_)
    , mu(mu_)
    , b(b_)
    , sigma(sigma_)
    , f(f_)
    //MY_OPTION, if I use MPI
    #ifdef USE_MPI
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, mpi_rank == 0)
    #endif
  {}

  #ifdef D_D
  // Apply Dirichlet conditions on the interface with another Poisson2D problem.
  void
  apply_interface_dirichlet(const DD &other);
  // Apply Neumann conditions on the interface with another Poisson2D problem.
  void
  apply_interface_neumann(DD &other);
  // Get the solution vector.
  const Vector<double> &
  get_solution() const
  {
    return solution;
  }
  // Apply relaxation.
  void
  apply_relaxation(const Vector<double> &old_solution, const double &lambda);
  #endif

  //MY_OPTION, if time dependant
  #ifdef TIME_DEPENDANT
  // Run the time-dependent simulation.
  void 
  run();
  #endif

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output(int num) const;

  // Compute the error against a given exact solution.
  double
  compute_error(const VectorTools::NormType &norm_type,
                const Function<dim>         &exact_solution) const;

protected:

  #ifdef D_D
  // Build an interface map, that is construct a map that to each DoF on the
  // interface for this subproblem associates the corresponding interface DoF on
  // the other subdomain.
  std::map<types::global_dof_index, types::global_dof_index>
  compute_interface_map(const DD &other) const;
  // ID of current subdomain (0 or 1).
  const unsigned int subdomain_id;
    // Support points.
  std::map<types::global_dof_index, Point<dim>> support_points;
  #endif

  //MY_OPTION_1
  // Number of elements.
  #ifndef EXTERNAL_MESH
  const unsigned int N_el;
  #else
  //MY_OPTION_2
  // Name of the mesh.
  const std::string mesh_file_name;
  #endif

  //MY_OPTION, if time dependant
  #ifdef TIME_DEPENDANT
  // Final time.
  const double T;
  // Theta parameter for the theta method.
  const double theta;
  // Time step.
  const double delta_t;
  // Current time.
  double time = 0.0;
  // Current timestep number.
  unsigned int timestep_number = 0;
  #endif

  // Polynomial degree.
  const unsigned int r;

  // Diffusion coefficient.
  std::function<double(const Point<dim> &)> mu;

  // Advection coefficient.
  std::function<Tensor<1,dim>(const Point<dim> &)> b;

  // Reaction coefficient.
  std::function<double(const Point<dim> &)> sigma;

  //MY_OPTION_1, if time independant
  #ifndef TIME_DEPENDANT
  // Forcing term.
  std::function<double(const Point<dim> &)> f;
  #else
  //MY_OPTION_2, if time dependant
  // Forcing term.
  std::function<double(const Point<dim> &, const double &)> f;
  #endif

  // MY_OPTION, if I use MPI
  #ifdef USE_MPI
  // Number of MPI processes.
  const unsigned int mpi_size;
  // Rank of the current MPI process.
  const unsigned int mpi_rank;
  #endif

  // MY_OPTION_1, if I use MPI
  #ifdef USE_MPI
  // Triangulation. The parallel::fullydistributed::Triangulation class manages
  // a mesh that is completely distributed across all MPI processes (i.e. each
  // process only stores its own locally relevant cells).
  parallel::fullydistributed::Triangulation<dim> mesh;
  #else
  //MY_OPTION_2, if I DON'T use MPI
  // Triangulation.
  Triangulation<dim> mesh;
  #endif

  // Finite element space.
  //
  // We use a unique_ptr here so that we can choose the type and degree of the
  // finite elements at runtime (the degree is a constructor parameter).
  //
  // The class FiniteElement<dim> is an abstract class from which all types of
  // finite elements implemented by deal.ii inherit. Using the abstract class
  // makes it very easy to switch between different types of FE space among the
  // many that deal.ii provides.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  //
  // We use a unique_ptr here so that we can choose the type and order of the
  // quadrature formula at runtime (the order is a constructor parameter).
  std::unique_ptr<Quadrature<dim>> quadrature;

  // MY_OPTION, if I have Neumann boundary conditions
  #ifdef NEUMANN
  // Quadrature formula for boundary integrals.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;
  #endif

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // MY_OPTION 1, if I use MPI
  #ifdef USE_MPI
  // System matrix.
  TrilinosWrappers::SparseMatrix system_matrix;
  // System right-hand side.
  TrilinosWrappers::MPI::Vector system_rhs;
  // System solution.
  TrilinosWrappers::MPI::Vector solution;
  // Output stream for process 0.
  ConditionalOStream pcout;
  // Locally owned DoFs for current process.
  IndexSet locally_owned_dofs;
  #else
  //MY_OPTION_2, if I DON'T use MPI
  // Sparsity pattern.
  SparsityPattern sparsity_pattern;
  // System matrix.
  SparseMatrix<double> system_matrix;
  // System right-hand side.
  Vector<double> system_rhs;
  // System solution.
  Vector<double> solution;
  #endif
};