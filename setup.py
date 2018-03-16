from sys import version_info

from setuptools import setup

install_requires = ['qualname',
                    'future']

tests_require = []

if version_info[0] == 2:
    tests_require += ['chainmap']

if version_info[:2] == (2, 6):
    install_requires += ['importlib']
    tests_require += ['ordereddict',
                      'counter']

if version_info[:2] == (2, 7) or version_info[:2] >= (3, 4):
    tests_require += ['numpy',
                      'Django' + '<2' * (version_info[0] == 2)]


setup(name='cheap_repr',
      version='0.1.4',
      description='Better version of repr/reprlib for short, cheap string representations.',
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ],
      url='http://github.com/alexmojaki/cheap_repr',
      author='Alex Hall',
      author_email='alex.mojaki@gmail.com',
      license='MIT',
      packages=['cheap_repr'],
      install_requires=install_requires,
      tests_require=tests_require,
      test_suite='tests',
      zip_safe=False)
