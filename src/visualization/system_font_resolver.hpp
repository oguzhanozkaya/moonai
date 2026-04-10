#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace moonai {

struct SystemFontResolution {
  std::filesystem::path path;
  std::string source;
};

std::optional<SystemFontResolution> resolve_system_monospace_font();

} // namespace moonai
