project('vs-fidelityfx', 'cpp',
  version: '1.0.2',
  default_options: ['cpp_std=c++20', 'warning_level=2', 'buildtype=release', 'b_lto=true']
)

# Dependencies
cpp = meson.get_compiler('cpp')
math_dep = cpp.find_library('m', required: false)

# Include directories
inc_dirs = include_directories('include')

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

# Enable SSE4.1 for faster float↔int conversions and blending
if cpp.has_argument('-msse4.1')
  add_project_arguments('-msse4.1', language: 'cpp')
endif

# Fast math (no NaN/Inf checks needed — pixel values are well-defined)
if cpp.has_argument('-ffast-math')
  add_project_arguments('-ffast-math', language: 'cpp')
endif

# Platform-specific flags
if host_machine.system() == 'windows'
  add_project_arguments('-D_CRT_SECURE_NO_WARNINGS', language: 'cpp')
endif

# Build shared library
if cpp.get_id() == 'gcc'
  link_args = ['-static-libgcc', '-static-libstdc++', '-Wl,-Bstatic', '-lpthread', '-Wl,-Bdynamic']
else
  link_args = []
endif

shared_library('vs_fidelityfx',
  sources,
  include_directories: inc_dirs,
  dependencies: [math_dep],
  link_args: link_args,
  name_prefix: '',
  install: true,
  install_dir: get_option('libdir')
)
