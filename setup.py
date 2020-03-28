from setuptools import setup,find_packages

about = {}
with open("medicalai/__about__.py") as fp:
    exec(fp.read(), about)


with open('README.md') as readme_file:
	readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()
	
setup(name='medicalai',
      version=about['__version__'],
      description='Medical-AI is a AI framework specifically for Medical Applications',
      url='https://github.com/aibharata/medicalAI',
      author=about['__author__'],
      author_email='contact@aibharata.com',
      license=about['__license__'],
	  install_requires=['pandas','tensorflow','numpy', 'matplotlib', 'plotly', 'pandas', 'seaborn', 'sklearn'],
      packages=find_packages(),
	  include_package_data=True,
	  package_data={
	  '': ['*.pyd',
			#'*.pyc', 
			'*.h5', '*.json','*.txt' ],
	  },
	  long_description=readme + history,
      classifiers=[
          'Development Status :: 1 - Development/Stable',
          'Intended Audience :: Science/Research',
          'Topic :: General/Engineering',
          'License :: OSI Approved :: Apache License',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      zip_safe=True,
	  python_requires='>=3.5, <3.8',
      extras_require={
          'test': ['pytest'],
      },
	  )