#pragma once

#include <string>

namespace slm::sheaf {

/// Metadata for a cohomology-triggered annotation.
struct Annotation {
    std::string superseded_text;   // text of the old (conflicting) node
    std::string superseding_text;  // text of the new node
    float delta_norm{0.0f};        // ||δ|| value
};

/// Format an annotation as an HTML comment for injection into active.md.
/// Output: <!-- cohomology: superseded "old" by "new", δ-norm=0.73 -->
std::string format_annotation(const Annotation& ann);

} // namespace slm::sheaf
