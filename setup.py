from setuptools import find_packages,setup
from typing import List

HYPER_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:        #this will take path in string format and return a list of strings
    '''
    this function will return the list of the requirments
    '''
    requirements=[]  #empty list
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()          #[numpy\n,pandas\n,seaborn\n,-e .] this will look like this
        requirements=[req.replace("\n","") for req in requirements]

        if HYPER_E_DOT in requirements:
            requirements.remove(HYPER_E_DOT)
    return requirements



setup(
    name='mlops project',
    version='0.0.1',
    author='vinay',
    author_email='vinayeditz0018@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)