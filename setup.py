from setuptools import find_packages, setup
from typing import List

FILENAME = 'requirements.txt'
HYPEN_E = '-e .'

def get_requirement(file_path: str) -> List[str]:
    '''
    this function will return the list of required packages for the project
    '''
    requirement = []
    with open(file_path) as file_obj:
       requirement = file_obj.readlines()
       requirement = [req.replace("\n", "") for req in requirement]

       if HYPEN_E in requirement:
           requirement.remove(HYPEN_E)

    return requirement

setup(name = 'student-performance-analysis', 
      version = '0.0.1', 
      description = 'student performance analysis using machine learning', 
      author = 'Jignesh Raiyani', 
      author_email = 'jignesh.raiyani1@gmail.com', 
      url = '', 
      download_url = '', 
      packages = find_packages(),
      license = '',
      install_requires = get_requirement(FILENAME))