#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void print () {
    std::cout << "Hello World!" << std::endl;
}

PYBIND11_MODULE (module_name, handle) {
    handle.doc() = "This is the module docs";
    handle.def("foo", &print);
}