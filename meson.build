project('vs-fidelityfx', 'cpp',
  version: '1.0.0',
  default_options: ['cpp_std=c++11', 'warning_level=2', 'buildtype=release']
)

# Dependencies
cpp = meson.get_compiler('cpp')
math_dep = cpp.find_library('m', required: false)

# Include directories
inc_dirs = include_directories('include', 'fidelityfx')

# Source files
sources = [
  'src/vs_fidelityfx.cpp',
  'src/fsr1_easu.cpp',
  'src/fsr1_rcas.cpp',
  'src/pixel_access.cpp',
  'src/lens_chromatic_aberration.cpp',
  'src/lens_vignette.cpp',
  'src/lens_grain.cpp'
]

# Compiler flags
add_project_arguments('-DFFX_CPU', language: 'cpp')

# Enable SSE2 on x86 (on x86_64 it's always available)
if host_machine.cpu_family() == 'x86'
  add_project_arguments('-msse2', language: 'cpp')
endif

# Platform-specific flags
if host_machine.system() == 'windows'
  add_project_arguments('-D_CRT_SECURE_NO_WARNINGS', language: 'cpp')
endif

# Build shared library
shared_library('vs_fidelityfx',
  sources,
  include_directories: inc_dirs,
  dependencies: [math_dep],
  name_prefix: '',
  install: true,
  install_dir: get_option('libdir')
)
