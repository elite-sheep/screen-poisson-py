#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "Defs.hpp"
#include "Solver.hpp"

namespace nb = nanobind;
using namespace nb::literals;

using namespace poisson;

using RGBImage = nb::ndarray<float, nb::shape<-1, -1, 3>, nb::device::cpu>;

NB_MODULE(screenpoissonpy, m) {
    nb::class_<poisson::SolverParams>(std::move(m), "PoissonParams")
        .def(nb::init<>())
        .def("setConfigPreset", &poisson::SolverParams::setConfigPreset)
        .def_rw("brightness", &poisson::SolverParams::brightness);

    nb::class_<poisson::Solver>(std::move(m), "PoissonSolver")
        .def(nb::init<poisson::SolverParams>())
        .def("solve", &poisson::Solver::solveIndirect)
        .def("load", [](poisson::Solver& self, RGBImage primal, RGBImage dx, RGBImage dy) {
            if (primal.shape(0) != dx.shape(0) || 
                primal.shape(1) != dx.shape(1) || 
                primal.shape(0) != dy.shape(0) || 
                primal.shape(1) != dy.shape(1)) {
                throw std::runtime_error("Images must have the same dimensions");
            }

            self.m_size = poisson::Vec2i(primal.shape(0), primal.shape(1));
            Vec3f* primalPtr = new Vec3f[primal.shape(0) * primal.shape(1)];
            Vec3f* dxPtr = new Vec3f[primal.shape(0) * primal.shape(1)];
            Vec3f* dyPtr = new Vec3f[primal.shape(0) * primal.shape(1)];
            for (int i = 0; i < self.m_size.x; ++i)
            {
                for (int j = 0; j < self.m_size.y; ++j)
                {
                    primalPtr[i * self.m_size.y + j] = Vec3f(primal(i, j, 0), primal(i, j, 1), primal(i, j, 2));
                    dxPtr[i * self.m_size.y + j] = Vec3f(dx(i, j, 0), dx(i, j, 1), dx(i, j, 2));
                    dyPtr[i * self.m_size.y + j] = Vec3f(dy(i, j, 0), dy(i, j, 1), dy(i, j, 2));
                }
            }

            self.m_throughput[0] = primalPtr;
            self.m_dx[0] = dxPtr;
            self.m_dy[0] = dyPtr;
        })
        .def("setupBackend", &poisson::Solver::setupBackend)
        .def("getFinalImage", [](poisson::Solver& self) {
            auto img = self.getFinalImage();
            float* result = new float[self.m_size.x * self.m_size.y * 3];
            int index = 0;
            for (int i = 0; i < self.m_size.x; ++i)
            {
                for (int j = 0; j < self.m_size.y; ++j)
                {
                    result[index++] = img[i * self.m_size.y + j].x;
                    result[index++] = img[i * self.m_size.y + j].y;
                    result[index++] = img[i * self.m_size.y + j].z;
                }
            }
            self.m_backend->unmap(self.m_r, (void*)img, false);

            return nb::ndarray<nb::numpy, float, nb::ndim<3>>(result, {(size_t)self.m_size.x, (size_t)self.m_size.y, 3}).cast();
        });
}