#pragma once

#include <cstdint>
#include <vector>
#include <sheaf/neighborhood.hpp>

namespace slm::sheaf {

struct CoboundaryResult {
    float norm{0.0f};
    std::vector<uint32_t> conflicting;
};

class CoboundaryOperator {
public:
    CoboundaryResult compute_local(
        const Neighborhood& hood,
        float threshold
    ) const;
};

} // namespace slm::sheaf
