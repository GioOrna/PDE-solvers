# PDE-solvers
A collection of some general code for the resolution of partial differential equations.

This code has been developed as an aid for writing code fast for the course of Partial Differantial Equations and uses the deal.II cpp library.

The code can be customized for specific equations by commenting out the #define in the .hpp file, also the customizable parts are preceded by the //TO_FILL comment. 

The Elliptic/Parabolic solver also has the option of using or not MPI and Domain Decomposition (the latter is not implemented in Parabolic (time-dependant) equations), and permits to choose an external mesh or generate one. Since there are a lot of options it is not well written (i didn't really have much time) but it should work.

The Steady/Unsteady Stokes solver only permits to choose between the two types of problem, the Steady solver has been fairly tested and should be correct, the Unsteady one has NOT been tested against a known problem so it could be NOT correct. 