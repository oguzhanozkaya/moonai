#pragma once

#include <cassert>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

// Profiler macros - minimal header for use in other translation units
#ifndef MOONAI_BUILD_PROFILER

#define MOONAI_PROFILE_SCOPE(event_name) ((void)0)

#else

namespace moonai {
namespace profiler {

// Forward declarations
class Profiler;
struct ScopeNode;

class ScopedTimer {
public:
  explicit ScopedTimer(const char *event_name);
  ~ScopedTimer();

private:
  const char *event_name_;
};

} // namespace profiler
} // namespace moonai

#define MOONAI_PROFILE_SCOPE(event_name)                                       \
  ::moonai::profiler::ScopedTimer _moonai_scoped_timer(event_name)

#endif
