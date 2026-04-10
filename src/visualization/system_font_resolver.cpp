#include "visualization/system_font_resolver.hpp"

#include <array>

#if defined(__linux__)
#include <fontconfig/fontconfig.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <dwrite.h>
#include <wrl/client.h>

#include <vector>
#endif

namespace moonai {
namespace {

#if defined(__linux__)

std::optional<SystemFontResolution> resolve_linux_font_family(const char *family_name) {
  FcPattern *pattern = FcPatternCreate();
  if (!pattern) {
    return std::nullopt;
  }

  FcPatternAddString(pattern, FC_FAMILY, reinterpret_cast<const FcChar8 *>(family_name));
  FcConfigSubstitute(nullptr, pattern, FcMatchPattern);
  FcDefaultSubstitute(pattern);

  FcResult result = FcResultNoMatch;
  FcPattern *match = FcFontMatch(nullptr, pattern, &result);
  FcPatternDestroy(pattern);
  if (!match) {
    return std::nullopt;
  }

  FcChar8 *resolved_file = nullptr;
  const FcResult file_result = FcPatternGetString(match, FC_FILE, 0, &resolved_file);
  std::optional<SystemFontResolution> resolution = std::nullopt;
  if (file_result == FcResultMatch && resolved_file != nullptr) {
    std::filesystem::path file_path(reinterpret_cast<const char *>(resolved_file));
    if (!file_path.empty() && std::filesystem::exists(file_path)) {
      resolution = SystemFontResolution{file_path, std::string("fontconfig:") + family_name};
    }
  }

  FcPatternDestroy(match);
  return resolution;
}

std::optional<SystemFontResolution> resolve_linux_monospace_font() {
  if (!FcInit()) {
    return std::nullopt;
  }

  constexpr std::array<const char *, 4> kCandidates = {
      "monospace",
      "DejaVu Sans Mono",
      "Noto Sans Mono",
      "Liberation Mono",
  };

  for (const char *candidate : kCandidates) {
    if (const auto resolved = resolve_linux_font_family(candidate)) {
      return resolved;
    }
  }

  return std::nullopt;
}

#elif defined(_WIN32)

using Microsoft::WRL::ComPtr;

std::optional<std::filesystem::path> local_font_path_from_file(const ComPtr<IDWriteFontFile> &font_file) {
  const void *reference_key = nullptr;
  UINT32 reference_key_size = 0;
  if (FAILED(font_file->GetReferenceKey(&reference_key, &reference_key_size)) || reference_key == nullptr) {
    return std::nullopt;
  }

  ComPtr<IDWriteFontFileLoader> loader;
  if (FAILED(font_file->GetLoader(loader.GetAddressOf())) || !loader) {
    return std::nullopt;
  }

  ComPtr<IDWriteLocalFontFileLoader> local_loader;
  if (FAILED(loader.As(&local_loader)) || !local_loader) {
    return std::nullopt;
  }

  UINT32 file_path_len = 0;
  if (FAILED(local_loader->GetFilePathLengthFromKey(reference_key, reference_key_size, &file_path_len)) ||
      file_path_len == 0) {
    return std::nullopt;
  }

  std::wstring file_path(file_path_len + 1, L'\0');
  if (FAILED(
          local_loader->GetFilePathFromKey(reference_key, reference_key_size, file_path.data(), file_path_len + 1))) {
    return std::nullopt;
  }

  if (!file_path.empty() && file_path.back() == L'\0') {
    file_path.pop_back();
  }

  if (file_path.empty()) {
    return std::nullopt;
  }

  return std::filesystem::path(file_path);
}

std::optional<SystemFontResolution> resolve_windows_font_family(const ComPtr<IDWriteFontCollection> &collection,
                                                                const wchar_t *family_name, const char *source_name) {
  UINT32 family_index = 0;
  BOOL family_exists = FALSE;
  if (FAILED(collection->FindFamilyName(family_name, &family_index, &family_exists)) || !family_exists) {
    return std::nullopt;
  }

  ComPtr<IDWriteFontFamily> family;
  if (FAILED(collection->GetFontFamily(family_index, family.GetAddressOf())) || !family) {
    return std::nullopt;
  }

  ComPtr<IDWriteFont> font;
  if (FAILED(family->GetFirstMatchingFont(DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STRETCH_NORMAL,
                                          DWRITE_FONT_STYLE_NORMAL, font.GetAddressOf())) ||
      !font) {
    return std::nullopt;
  }

  ComPtr<IDWriteFontFace> face;
  if (FAILED(font->CreateFontFace(face.GetAddressOf())) || !face) {
    return std::nullopt;
  }

  UINT32 file_count = 0;
  if (FAILED(face->GetFiles(&file_count, nullptr)) || file_count == 0) {
    return std::nullopt;
  }

  std::vector<IDWriteFontFile *> raw_files(file_count, nullptr);
  if (FAILED(face->GetFiles(&file_count, raw_files.data()))) {
    return std::nullopt;
  }

  for (IDWriteFontFile *raw_file : raw_files) {
    ComPtr<IDWriteFontFile> font_file;
    font_file.Attach(raw_file);
    if (!font_file) {
      continue;
    }

    const auto font_path = local_font_path_from_file(font_file);
    if (!font_path || font_path->empty() || !std::filesystem::exists(*font_path)) {
      continue;
    }

    return SystemFontResolution{*font_path, std::string("directwrite:") + source_name};
  }

  return std::nullopt;
}

std::optional<SystemFontResolution> resolve_windows_monospace_font() {
  ComPtr<IDWriteFactory> factory;
  const HRESULT create_factory_result = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
                                                            reinterpret_cast<IUnknown **>(factory.GetAddressOf()));
  if (FAILED(create_factory_result) || !factory) {
    return std::nullopt;
  }

  ComPtr<IDWriteFontCollection> collection;
  if (FAILED(factory->GetSystemFontCollection(collection.GetAddressOf())) || !collection) {
    return std::nullopt;
  }

  struct FamilyCandidate {
    const wchar_t *name;
    const char *label;
  };

  constexpr std::array<FamilyCandidate, 4> kCandidates = {
      FamilyCandidate{L"Consolas", "Consolas"},
      FamilyCandidate{L"Cascadia Mono", "Cascadia Mono"},
      FamilyCandidate{L"Courier New", "Courier New"},
      FamilyCandidate{L"Lucida Console", "Lucida Console"},
  };

  for (const auto &candidate : kCandidates) {
    if (const auto resolved = resolve_windows_font_family(collection, candidate.name, candidate.label)) {
      return resolved;
    }
  }

  return std::nullopt;
}

#endif

} // namespace

std::optional<SystemFontResolution> resolve_system_monospace_font() {
#if defined(__linux__)
  return resolve_linux_monospace_font();
#elif defined(_WIN32)
  return resolve_windows_monospace_font();
#else
  return std::nullopt;
#endif
}

} // namespace moonai
