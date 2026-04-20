#include <sheaf/annotation.hpp>
#include <sstream>
#include <iomanip>

namespace slm::sheaf {

namespace {

std::string escape_quotes(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    for (char c : text) {
        if (c == '"') {
            result += "\\\"";
        } else {
            result += c;
        }
    }
    return result;
}

} // namespace

std::string format_annotation(const Annotation& ann) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "<!-- cohomology: superseded \""
        << escape_quotes(ann.superseded_text)
        << "\" by \""
        << escape_quotes(ann.superseding_text)
        << "\", δ-norm=" << ann.delta_norm
        << " -->";
    return oss.str();
}

} // namespace slm::sheaf
