#from distutils.core import setup
from setuptools import setup,find_packages

about = {}
with open("medicalai/__about__.py") as fp:
    exec(fp.read(), about)


with open('README.md') as readme_file:
	readme = readme_file.read()
	
setup(name='medicalai',
      #packages = ['medicalai'],
      version=about['__version__'],
      description='Medical-AI is a AI framework specifically for Medical Applications',
      url='https://github.com/aibharata/medicalAI',
      download_url = 'https://github.com/aibharata/medicalAI/archive/v1_04.tar.gz',
      keywords = ['AI Framework', 'Medical AI', 'Tensorflow', 'radiology AI'],
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
	  long_description=readme,
      long_description_content_type='text/markdown',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'License :: OSI Approved :: Apache Software License',
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
