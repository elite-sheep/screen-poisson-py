#include <nanobind/nanobind.h>

int add(int a, int b) {
    return a + b;
}

NB_MODULE(screen_poisson_py, m) {
    m.def("add", &add);
}