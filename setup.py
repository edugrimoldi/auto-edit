from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='autoedit',
      #version="0.0.1",
      description="Video Auto-Edit Model (train locally)",
      license="MIT",
      author="Gameplay VAE team",
      #author_email="contact@email.com",
      url="https://github.com/edugrimoldi/auto-edit",
      install_requires=requirements,
      packages=find_packages(),
      #test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
