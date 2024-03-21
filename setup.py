# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""setup.py for Scenic.

Install for development:

  pip intall -e . .[testing]
"""

import os
import urllib.request

from setuptools import Command
from setuptools import find_packages
from setuptools import setup
from setuptools.command import install

SIMCLR_DIR = "simclr/tf2"
DATA_UTILS_URL = "https://raw.githubusercontent.com/google-research/simclr/master/tf2/data_util.py"


class DownloadSimCLRAugmentationCommand(Command):
  """Downloads SimCLR data_utils.py as it's not built into an egg."""
  description = __doc__
  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    build_cmd = self.get_finalized_command("build")
    dist_root = os.path.realpath(build_cmd.build_lib)
    output_dir = os.path.join(dist_root, SIMCLR_DIR)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "data_util.py")
    downloader = urllib.request.URLopener()
    downloader.retrieve(DATA_UTILS_URL, output_path)


class InstallCommand(install.install):

  def run(self):
    self.run_command("simclr_download")
    install.install.run(self)


install_requires_projects = [
    "ott-jax>=0.2.0",
    "sklearn",
    "lingvo==0.12.6",
    "seaborn>=0.11.2",
    "dmvr @ git+https://github.com/google-deepmind/dmvr.git",
]

install_requires_core = [
    "absl-py>=1.0.0",
    "numpy>=1.12",
    "jax>=0.4.3",
    "jaxlib>=0.4.3",
    "flax>=0.4.0",
    "ml-collections>=0.1.1",
    "tensorflow>=2.7",
    "immutabledict>=2.2.1",
    "clu>=0.0.6",
    "tensorflow-datasets",
    "optax @ git+https://github.com/google-deepmind/optax.git@main",
]

tests_require = [
    "pytest",
    "shapely",
] + install_requires_projects

setup(
    name="scenic",
    version="0.0.1",
    description=("A Jax Library for Computer Vision Research and Beyond."),
    author="Scenic Authors",
    author_email="no-reply@google.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/google-research/scenic",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires_core,
    cmdclass={
        "simclr_download": DownloadSimCLRAugmentationCommand,
        "install": InstallCommand,
    },
    tests_require=tests_require,
    extras_require={
        "testing": tests_require,
    },
    classifiers=[
        "Development Status :: 1 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="Scenic",
)
