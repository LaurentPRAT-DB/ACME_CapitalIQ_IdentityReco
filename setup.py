# Copyright 2026 Laurent Prat
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

"""
Setup script for entity-matching-capitaliq package
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="entity-matching-capitaliq",
    version="1.0.0",
    description="GenAI-powered entity matching for S&P Capital IQ identity reconciliation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Laurent Prat",
    author_email="laurent.prat@databricks.com",
    url="https://github.com/LaurentPRAT-DB/MET_CapitalIQ_identityReco",
    license="Apache License 2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyspark>=3.4.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "databricks-sdk>=0.18.0",
        "mlflow>=2.9.0",
        "gliner>=0.1.0",
        "faiss-cpu>=1.7.4",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "pyarrow>=12.0.0",
        "delta-spark>=2.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Financial",
    ],
)
