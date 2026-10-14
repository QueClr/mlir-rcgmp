# Locate the GMP C API, including its public mpn functions. GMP_ROOT and the
# standard CMake search paths may be used for non-system installations.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_GMP QUIET gmp)
endif()

find_path(GMP_INCLUDE_DIR NAMES gmp.h HINTS ${PC_GMP_INCLUDE_DIRS})
find_library(GMP_LIBRARY NAMES gmp HINTS ${PC_GMP_LIBRARY_DIRS})

if(GMP_INCLUDE_DIR)
  file(STRINGS "${GMP_INCLUDE_DIR}/gmp.h" _gmp_version_lines
       REGEX "^#define[ \t]+__GNU_MP_VERSION(_MINOR|_PATCHLEVEL)?[ \t]+[0-9]+")
  foreach(_component MAJOR MINOR PATCHLEVEL)
    if(_component STREQUAL "MAJOR")
      set(_macro "__GNU_MP_VERSION")
    else()
      set(_macro "__GNU_MP_VERSION_${_component}")
    endif()
    string(REGEX MATCH "${_macro}[ \t]+([0-9]+)" _match "${_gmp_version_lines}")
    set(_gmp_${_component} "${CMAKE_MATCH_1}")
  endforeach()
  set(GMP_VERSION "${_gmp_MAJOR}.${_gmp_MINOR}.${_gmp_PATCHLEVEL}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP
  REQUIRED_VARS GMP_LIBRARY GMP_INCLUDE_DIR
  VERSION_VAR GMP_VERSION)

if(GMP_FOUND AND NOT TARGET GMP::GMP)
  add_library(GMP::GMP UNKNOWN IMPORTED)
  set_target_properties(GMP::GMP PROPERTIES
    IMPORTED_LOCATION "${GMP_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}")
endif()

mark_as_advanced(GMP_INCLUDE_DIR GMP_LIBRARY)
