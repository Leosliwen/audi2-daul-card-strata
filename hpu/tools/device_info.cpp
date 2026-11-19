#include "strata/hpu/runtime.hpp"
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 1 && (argc != 3 || std::string(argv[1]) != "--pci"))
            throw std::invalid_argument("usage: strata-hpu-info [--pci DOMAIN:BUS:DEVICE.FUNCTION]");
        strata::hpu::Device device(std::nullopt, argc == 3 ? argv[2] : "");
        const auto info = device.info();
        std::cout << "device=" << info.name << "\ndriver=" << info.driver
                  << "\nfree_bytes=" << info.free_bytes << "\ntotal_bytes=" << info.total_bytes << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
