from sys import version_info, version

from setuptools import setup

install_requires = []

if version_info[0] == 2:
    tests_require = ['Django<2',
                     'chainmap']
    install_requires += ['qualname']
elif version_info[:2] == (3, 5):
    tests_require = ['Django<3']
else:
    tests_require = ['Django']

if 'pypy' not in version.lower():
    if version_info[0] == 2:
        tests_require += ['pandas>=0.24.2,<0.25']
        tests_require += ['numpy>=1.16.3,<1.17']
    elif version_info[:2] == (3, 5):
        tests_require += ['pandas>=0.24.2,<0.26']
        tests_require += ['numpy>=1.16.3,<1.19']
    else:
        tests_require += ['pandas>=0.24.2']
        tests_require += ['numpy>=1.16.3']

tests_require += ['pytest']

print(version_info, tests_require)


setup(name='cheap_repr',
      version='0.4.4',
      description='Better version of repr/reprlib for short, cheap string representations.',
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: Implementation :: CPython',
          'Programming Language :: Python :: Implementation :: PyPy',
          'Operating System :: OS Independent',
          'Intended Audience :: Developers',
      ],
      url='http://github.com/alexmojaki/cheap_repr',
      author='Alex Hall',
      author_email='alex.mojaki@gmail.com',
      license='MIT',
      packages=['cheap_repr'],
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require={'tests': tests_require},
      test_suite='tests',
      zip_safe=False)
