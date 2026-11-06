project(
  'micrograd',
  'c',
  meson_version : '>= 1.3.0',
  version : '0.1',
  default_options : ['warning_level=3', 'c_std=c11'],
)

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required : false)
openblas_dep = dependency('openblas', required : true)

dependencies = [
  m_dep,
  openblas_dep,
]

inc = include_directories('include')

sources = [
  'src/main.c',
  'src/engine.c',
  'src/nn.c',
  'src/infer.c',
]

exe = executable(
  'micrograd',
  sources,
  include_directories : inc,
  dependencies : dependencies,
  install : true,
)

test('basic', exe)
