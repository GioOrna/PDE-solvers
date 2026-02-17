#include "Scalar_Elliptic_Parabolic.hpp"

void DD::setup(){
    #ifdef EXTERNAL_MESH
    #ifndef USE_MPI
    // Read the mesh from file.
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh);
    std::ifstream mesh_file(mesh_file_name);
    grid_in.read_msh(mesh_file);
    #else
    // First, we read the mesh from file into a serial (i.e. not parallel)
    // triangulation.
    Triangulation<dim> mesh_serial;
    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(mesh_serial);

      std::ifstream mesh_file(mesh_file_name);
      grid_in.read_msh(mesh_file);
    }
    // Then, we copy the serial mesh into the parallel one.
    {
      GridTools::partition_triangulation(mpi_size, mesh_serial);
      const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
      mesh.create_triangulation(construction_data);
    }
    #endif
    #else
    //TO_FILL
    double a = 0.0;
    double b = 1.0;
    GridGenerator::subdivided_hyper_cube(mesh, N_el, a, b);
    #endif

    fe = std::make_unique<FE_SimplexP<dim>>(r);
    std::cout << "  Degree                     = " << fe->degree << std::endl;
    std::cout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);
    std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;

    #ifdef D_D
        // Compute support points for the DoFs.
    FE_SimplexP<dim> fe_linear(1);
    MappingFE        mapping(fe_linear);
    support_points = DoFTools::map_dofs_to_support_points(mapping, dof_handler);
    #endif

    //MY_OPTION, if parallel using MPI
    #ifdef USE_MPI
    // We retrieve the set of locally owned DoFs, which will be useful when
    // initializing linear algebra classes.
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    #endif

    #ifndef USE_MPI
    // We first initialize a "sparsity pattern", i.e. a data structure that
    // indicates which entries of the matrix are zero and which are different
    // from zero. To do so, we construct first a DynamicSparsityPattern (a
    // sparsity pattern stored in a memory- and access-inefficient way, but
    // fast to write) and then convert it to a SparsityPattern (which is more
    // efficient, but cannot be modified).
    std::cout << "  Initializing the sparsity pattern" << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    // Then, we use the sparsity pattern to initialize the system matrix
    std::cout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity_pattern);

    // Finally, we initialize the right-hand side and solution vectors.
    std::cout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(dof_handler.n_dofs());
    std::cout << "  Initializing the solution vector" << std::endl;
    solution.reinit(dof_handler.n_dofs());
    #else
    // For the sparsity pattern, we use Trilinos' class, which manages some of
    // the inter-process communication.
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);

    // After initialization, we need to call compress, so that all processes
    // retrieve the information they need from the rows they own (i.e. the rows
    // corresponding to locally owned DoFs).
    sparsity.compress();

    pcout << "  Initializing the system matrix" << std::endl;
    // Since the sparsity pattern is partitioned by row, so will be the matrix.
    system_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    #endif
}

void DD::assemble(){
    // Construct the quadrature formula of the appopriate degree of exactness.
    // This formula integrates exactly the mass matrix terms (i.e. products of
    // basis functions).
    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);
    std::cout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;
    // FEValues instance. This object allows to compute basis functions, their
  // derivatives, the reference-to-current element mapping and its
  // derivatives on all quadrature points of all elements.
  FEValues<dim> fe_values(
    *fe,
    *quadrature,
    // Here we specify what quantities we need FEValues to compute on
    // quadrature points. For our test, we need:
    // - the values of shape functions (update_values);
    // - the derivative of shape functions (update_gradients);
    // - the position of quadrature points (update_quadrature_points);
    // - the quadrature weights (update_JxW_values).
    update_values | update_gradients | update_quadrature_points |
      update_JxW_values);

    // MY_OPTION, IF NEUMANN BOUNDARY CONDITION
  #ifdef NEUMANN
  quadrature_boundary = std::make_unique<QGaussSimplex<dim - 1>>(r + 1);
  // Since we need to compute integrals on the boundary for Neumann conditions,
  // we also need a FEValues object to compute quantities on boundary edges
  // (faces).
  FEFaceValues<dim> fe_values_boundary(*fe,
                                       *quadrature_boundary,
                                       update_values |
                                         update_quadrature_points |
                                         update_JxW_values);
  #endif

  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  // Local matrix and right-hand side vector. We will overwrite them for
  // each element within the loop.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  // We will use this vector to store the global indices of the DoFs of the
  // current element within the loop.
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs    = 0.0;

  //MY_OPTION, if time dependant
  #ifdef TIME_DEPENDANT
  // Evaluation of the old solution on quadrature nodes of current cell.
  std::vector<double> solution_old_values(n_q);
  // Evaluation of the gradient of the old solution on quadrature nodes of
  // current cell.
  std::vector<Tensor<1, dim>> solution_old_grads(n_q);
  #endif

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      //MY_OPTION, if in parallel using MPI
      #ifdef USE_MPI
      // If current cell is not locally owned, we skip it.
      if (!cell->is_locally_owned())
        continue;
      // On all other cells (which are owned by current process) we perform
      // assembly as usual.
      #endif

      // Reinitialize the FEValues object on current element. This
      // precomputes all the quantities we requested when constructing
      // FEValues (see the update_* flags above) for all quadrature nodes of
      // the current cell.
      fe_values.reinit(cell);

      // We reset the cell matrix and vector (discarding any leftovers from
      // previous element).
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      //MY_OPTION, if time dependant
      #ifdef TIME_DEPENDANT
      // Evaluate the old solution and its gradient on quadrature nodes.
      fe_values.get_function_values(solution, solution_old_values);
      fe_values.get_function_gradients(solution, solution_old_grads);
      #endif

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Here we assemble the local contribution for current cell and
          // current quadrature point, filling the local matrix and vector.
          const double mu_loc = mu(fe_values.quadrature_point(q));
          const Tensor<1, dim> b_loc = b(fe_values.quadrature_point(q));
          const double sigma_loc = sigma(fe_values.quadrature_point(q));

          #ifdef TIME_DEPENDANT
            const double f_old_loc = f(fe_values.quadrature_point(q), time - delta_t);
            const double f_new_loc = f(fe_values.quadrature_point(q), time);
          #else
            const double f_loc  = f(fe_values.quadrature_point(q));
          #endif

          // Here we iterate over *local* DoF indices.
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                //TO_FILL
                  #ifdef TIME_DEPENDANT
                    // Time derivative.
                    cell_matrix(i, j) += (1.0 / delta_t) *             //
                                        fe_values.shape_value(i, q) * //
                                        fe_values.shape_value(j, q) * //
                                        fe_values.JxW(q);
                    // Diffusion.
                    cell_matrix(i, j) +=
                      theta * mu_loc *                             //
                      scalar_product(fe_values.shape_grad(i, q),   //
                                    fe_values.shape_grad(j, q)) * //
                      fe_values.JxW(q);
                    // Advection (done by me check it)
                    cell_matrix(i, j) += theta * scalar_product(b_loc,                     //
                                        fe_values.shape_grad(i, q)) * //
                                        fe_values.shape_value(j, q) * //
                                        fe_values.JxW(q);
                    // Reaction (done by me check it)
                    cell_matrix(i, j) += theta * sigma_loc *                   //
                                          fe_values.shape_value(i, q) * //
                                          fe_values.shape_value(j, q) * //
                                          fe_values.JxW(q);
                  #else
                    // Diffusion
                    cell_matrix(i, j) += mu_loc *                     //
                                        fe_values.shape_grad(i, q) * //
                                        fe_values.shape_grad(j, q) * //
                                        fe_values.JxW(q);

                    // Advection
                    cell_matrix(i, j) += scalar_product(b_loc,                     //
                                        fe_values.shape_grad(i, q)) * //
                                        fe_values.shape_value(j, q) * //
                                        fe_values.JxW(q);

                    // Reaction
                    cell_matrix(i, j) += sigma_loc *                   //
                                        fe_values.shape_value(i, q) * //
                                        fe_values.shape_value(j, q) * //
                                        fe_values.JxW(q);
                  #endif
                }
              #ifdef TIME_DEPENDANT
                // Time derivative.
                cell_rhs(i) += (1.0 / delta_t) *             //
                             fe_values.shape_value(i, q) * //
                             solution_old_values[q] *      //
                             fe_values.JxW(q);
                // Diffusion.
                cell_rhs(i) -= (1.0 - theta) * mu_loc *                   //
                             scalar_product(fe_values.shape_grad(i, q), //
                                            solution_old_grads[q]) *    //
                             fe_values.JxW(q);
                // Advection (Done by me check it)
                cell_rhs(i) -= (1.0 - theta) * scalar_product(b_loc,                     //
                                        solution_old_grads[q]) * //
                                        fe_values.shape_value(i, q) * //
                                        fe_values.JxW(q);
                // Reaction. (Done by me check it)
                cell_rhs(i) -= (1.0 - theta) * sigma_loc *             //
                                        fe_values.shape_value(i, q) * //
                                        solution_old_values[q] * //
                                        fe_values.JxW(q);
                // Forcing term.
                cell_rhs(i) +=
                (theta * f_new_loc + (1.0 - theta) * f_old_loc) * //
                fe_values.shape_value(i, q) *                     //
                fe_values.JxW(q);
              #else
                // Forcing term
                cell_rhs(i) += f_loc *                       //
                             fe_values.shape_value(i, q) * //
                             fe_values.JxW(q);
              #endif
            }
        }
      
      // MY_OPTION IF NEUMANN BOUNDARY CONDITION
      #ifdef NEUMANN
      // If the cell is adjacent to the boundary...
      if (cell->at_boundary())
        {
          // ...we loop over its edges (referred to as faces in the deal.II
          // jargon).
          for (unsigned int face_number = 0; face_number < cell->n_faces();
               ++face_number)
            {
              // If current face lies on the boundary, and its boundary ID (or
              // tag) is that of one of the Neumann boundaries, we assemble the
              // boundary integral.
              if (cell->face(face_number)->at_boundary() &&
              //TO_FILL
                  (cell->face(face_number)->boundary_id() == 2 ||
                   cell->face(face_number)->boundary_id() == 3))
                {
                  fe_values_boundary.reinit(cell, face_number);

                  for (unsigned int q = 0; q < quadrature_boundary->size(); ++q)
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      //TO_FILL
                      //constant/function * base_function
                      //P.S. the point coordinates are fe_values_boundary.quadrature_point(q), so fe_values_boundary.quadrature_point(q)[0] is the x coordinate
                        cell_rhs(i) +=
                          1.0 * //
                          fe_values_boundary.shape_value(i, q) *      //
                          fe_values_boundary.JxW(q);
                    }
                }
            }
        }
      #endif

      // At this point the local matrix and vector are constructed: we need
      // to sum them into the global matrix and vector. To this end, we need
      // to retrieve the global indices of the DoFs of current cell.
      cell->get_dof_indices(dof_indices);

      // Then, we add the local matrix and vector into the corresponding
      // positions of the global matrix and vector.
      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

    // MY_OPTION, if parallel using MPI
    #ifdef USE_MPI
    // Each process might have written to some rows it does not own (for instance,
    // if it owns elements that are adjacent to elements owned by some other
    // process). Therefore, at the end of assembly, processes need to exchange
    // information: the compress method allows to do this.
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
    #endif

    // Dirichlet Boundary conditions.
    #ifdef DIRICHLET
  //
  // So far we assembled the matrix as if there were no Dirichlet conditions.
  // Now we want to replace the rows associated to nodes on which Dirichlet
  // conditions are applied with equations like u_i = b_i. We use deal.ii
  // functions to
  {
    // We construct a map that stores, for each DoF corresponding to a Dirichlet
    // condition, the corresponding value. E.g., if the Dirichlet condition is
    // u_i = b_i, the map will contain the pair (i, b_i).
    std::map<types::global_dof_index, double> boundary_values;

    // MY_OPTION_1 if we use custom Dirichlet boundary conditions
    // This object represents our boundary data as a real-valued function (that
    // always evaluates to zero). Other functions may require to implement a
    // custom class derived from dealii::Function<dim>.
    // this is a custom class for custom boundary: 
    //TO_FILL
    //FunctionG bc_function;
    // MY_OPTION_2, if we use standard Dirichlet boundary condition:
    // this is an example of a standard class:
    //TO_FILL
    Functions::ZeroFunction<dim> bc_function;
    // Then, we build a map that, for each boundary tag, stores a pointer to the
    // corresponding boundary function.
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    #ifdef D_D
    //TO_FILL
    if(subdomain_id==0){
        boundary_functions[0] = &bc_function;
        boundary_functions[2] = &bc_function;
        boundary_functions[3] = &bc_function;
    }
    else{
        boundary_functions[2] = &bc_function;
        boundary_functions[1] = &bc_function;
        boundary_functions[3] = &bc_function;
    }
    #else
    //TO_FILL
        boundary_functions[0] = &bc_function;
        boundary_functions[1] = &bc_function;
    #endif

    // interpolate_boundary_values fills the boundary_values map.
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    // Finally, we modify the linear system to apply the boundary conditions.
    // This replaces the equations for the boundary DoFs with the corresponding
    // u_i = 0 equations.
    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, true);
  }
  #endif
}

void DD::solve(){
    #ifndef USE_MPI

  //TO_FILL
  //MY_OPTION_1
  PreconditionSSOR preconditioner;
  preconditioner.initialize(
  system_matrix, PreconditionSSOR<SparseMatrix<double>>::AdditionalData(1.0));
  //TO_FILL
  //MY_OPTION_2
  //PreconditionIdentity preconditioner;

  // Here we specify the maximum number of iterations of the iterative solver,
  // and its absolute and relative tolerances.
  ReductionControl solver_control(/* maxiter = */ 1000,
                                  /* tolerance = */ 1.0e-16,
                                  /* reduce = */ 1.0e-6);

  // Since the system matrix is symmetric and positive definite, we solve the
  // system using the conjugate gradient method.
  SolverCG<Vector<double>> solver(solver_control);

  std::cout << "  Solving the linear system" << std::endl;
  // We use the identity preconditioner for now.
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  std::cout << "  " << solver_control.last_step() << " CG iterations"
            << std::endl;
    #else
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(
    system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  ReductionControl solver_control(/* maxiter = */ 10000,
                                  /* tolerance = */ 1.0e-16,
                                  /* reduce = */ 1.0e-6);

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  pcout << "  Solving the linear system" << std::endl;

  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
  #endif
}

void DD::output(int num) const{
  #ifndef USE_MPI
  #ifdef D_D
  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution, "solution");
  data_out.build_patches();

  const std::string output_file_name = "output-" +
                                       std::to_string(subdomain_id) + "-" +
                                       std::to_string(num) + ".vtk";
  std::ofstream output_file(output_file_name);
  data_out.write_vtk(output_file);
  #else
  std::cout << "===============================================" << std::endl;

  // The DataOut class manages writing the results to a file.
  DataOut<dim> data_out;

  // It can write multiple variables (defined on the same mesh) to a single
  // file. Each of them can be added by calling add_data_vector, passing the
  // associated DoFHandler and a name.
  data_out.add_data_vector(dof_handler, solution, "solution");

  // Once all vectors have been inserted, call build_patches to finalize the
  // DataOut object, preparing it for writing to file.
  data_out.build_patches();

  // Then, use one of the many write_* methods to write the file in an
  // appropriate format.
  const std::string output_file_name =
  //TO_FILL: + std::to_string() +
    "output" + std::to_string(num) + ".vtk";
  std::ofstream output_file(output_file_name);
  data_out.write_vtk(output_file);

  std::cout << "Output written to " << output_file_name << std::endl;

  std::cout << "===============================================" << std::endl;
  #endif
  #else
  pcout << "===============================================" << std::endl;

  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  // To correctly export the solution, each process needs to know the solution
  // DoFs it owns, and the ones corresponding to elements adjacent to the ones
  // it owns (the locally relevant DoFs, or ghosts). We create a vector to store
  // them.
  TrilinosWrappers::MPI::Vector solution_ghost(locally_owned_dofs,
                                               locally_relevant_dofs,
                                               MPI_COMM_WORLD);

  // This assignment performs the necessary communication so that the locally
  // relevant DoFs are received from other processes and stored inside
  // solution_ghost.
  solution_ghost = solution;

  // Then we build and fill the DataOut class as usual, using the vector with
  // ghosts.
  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution_ghost, "solution");

  // We also add a vector to represent the parallel partitioning of the mesh,
  // for the sake of visualization.
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::filesystem::path mesh_path(mesh_file_name);
  const std::string output_file_name = "output-" + mesh_path.stem().string();

  // Finally, we need to write in a format that supports parallel output. This
  // can be achieved in multiple ways (e.g. XDMF/H5). We choose VTU/PVTU files.
  data_out.write_vtu_with_pvtu_record(/* folder = */ "./",
                                      /* basename = */ output_file_name,
                                      /* index = */ num,
                                      MPI_COMM_WORLD);

  pcout << "Output written to " << output_file_name << std::endl;

  pcout << "===============================================" << std::endl;
  #endif
}

#ifdef TIME_DEPENDANT
void DD::run(){
    // Setup initial conditions.
  {
    setup();
    VectorTools::interpolate(dof_handler, FunctionU0(), solution);
    time            = 0.0;
    timestep_number = 0;
    // Output initial condition.
    output(timestep_number);
  }
  #ifdef USE_MPI
  pcout 
  #else
  std::cout
  #endif 
  << "===============================================" << std::endl;

  // Time-stepping loop.
  while (time < T - 0.5 * delta_t)
    {
      time += delta_t;
      ++timestep_number;

        #ifdef USE_MPI
        pcout 
        #else
        std::cout
        #endif << "Timestep " << std::setw(3) << timestep_number
            << ", time = " << std::setw(4) << std::fixed << std::setprecision(2)
            << time << " : ";

      assemble();
      solve();
      output(timestep_number);
    }
}
#endif

double
DD::compute_error(const VectorTools::NormType &norm_type,
                         const Function<dim>         &exact_solution) const
{
  // The error is an integral, and we approximate that integral using a
  // quadrature formula. To make sure we are accurate enough, we use a
  // quadrature formula with one node more than what we used in assembly.
  const QGaussSimplex<dim> quadrature_error(r + 2);

  FE_SimplexP<dim> fe_linear(1);
  MappingFE        mapping(fe_linear);
  // First we compute the norm on each element, and store it in a vector.
  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(mapping,      dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  // Then, we add out all the cells.
  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}

#ifdef D_D
void
DD::apply_interface_dirichlet(const DD &other)
{
  const auto interface_map = compute_interface_map(other);

  // We use the interface map to build a boundary values map for interface DoFs.
  std::map<types::global_dof_index, double> boundary_values;
  for (const auto &dof : interface_map)
    boundary_values[dof.first] = other.solution[dof.second];

  // Then, we apply those boundary values.
  MatrixTools::apply_boundary_values(
    boundary_values, system_matrix, solution, system_rhs, false);
}

void
DD::apply_interface_neumann(DD &other)
{
  const auto interface_map = compute_interface_map(other);

  // We assemble the interface residual of the other subproblem. Indeed,
  // directly computing the normal derivative of the solution on the other
  // subdomain has extremely poor accuracy. This is due to the fact that the
  // trace of the derivative has very low regularity. Therefore, we compute the
  // (weak) normal derivative as the residual of the system of the other
  // subdomain, excluding interface conditions.
  Vector<double> interface_residual;
  other.assemble();
  interface_residual = other.system_rhs;
  interface_residual *= -1;
  other.system_matrix.vmult_add(interface_residual, other.get_solution());

  // Then, we add the elements of the residual corresponding to interface DoFs
  // to the system rhs for current subproblem.
  for (const auto &dof : interface_map)
    system_rhs[dof.first] -= interface_residual[dof.second];
}

std::map<types::global_dof_index, types::global_dof_index>
DD::compute_interface_map(const DD &other) const
{
  // Retrieve interface DoFs on the current and other subdomain.
  IndexSet current_interface_dofs;
  IndexSet other_interface_dofs;

  if (subdomain_id == 0)
    {
      current_interface_dofs =
      //TO_FILL
      //boundary_id of the interface in the subdomain_0 and subdomain_1
        DoFTools::extract_boundary_dofs(dof_handler, ComponentMask(), {1});
      other_interface_dofs = DoFTools::extract_boundary_dofs(other.dof_handler,
                                                             ComponentMask(),
                                                             {0});
    }
  else
    {
      current_interface_dofs =
      //TO_FILL
    //boundary_id of the interface in the subdomain_1 and subdomain_0
        DoFTools::extract_boundary_dofs(dof_handler, ComponentMask(), {0});
      other_interface_dofs = DoFTools::extract_boundary_dofs(other.dof_handler,
                                                             ComponentMask(),
                                                             {1});
    }

  // For each interface DoF on current subdomain, we find the corresponding one
  // on the other subdomain.
  std::map<types::global_dof_index, types::global_dof_index> interface_map;
  for (const auto &dof_current : current_interface_dofs)
    {
      const Point<dim> &p = support_points.at(dof_current);

      types::global_dof_index nearest = *other_interface_dofs.begin();
      for (const auto &dof_other : other_interface_dofs)
        {
          if (p.distance_square(other.support_points.at(dof_other)) <
              p.distance_square(other.support_points.at(nearest)))
            nearest = dof_other;
        }

      interface_map[dof_current] = nearest;
    }

  return interface_map;
}

void
DD::apply_relaxation(const Vector<double> &old_solution,
                            const double         &lambda)
{
  solution *= lambda;
  solution.add(1.0 - lambda, old_solution);
}
#endif