import setuptools
from torch.utils.cpp_extension import BuildExtension, CppExtension

setuptools.setup(name='kernel.cpp', ext_modules=[CppExtension('kernel.cpp', ['kernel.cpp'])],
                 cmdclass={'build_ext': BuildExtension})
