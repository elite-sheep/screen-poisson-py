#include <nanobind/nanobind.h>

#include "Solver.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(screenpoissonpy, m) {
    nb::class_<poisson::SolverParams>(std::move(m), "PoissonParams")
        .def(nb::init<>())
        .def("setConfigPreset", &poisson::SolverParams::setConfigPreset)
        .def_rw("brightness", &poisson::SolverParams::brightness);

    nb::class_<poisson::Solver>(std::move(m), "PoissonSolver")
        .def(nb::init<poisson::SolverParams>())
        .def("solve", &poisson::Solver::solveIndirect);
}