from setuptools import setup, find_packages

setup(
    name="data-cleaning-env",
    version="1.0.0",
    description="Real-world data cleaning and validation environment for training AI agents",
    author="OpenEnv Community",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "pydantic==2.5.0",
        "openai>=2.7.2",
        "pandas==2.1.3",
        "numpy==1.26.2",
        "openenv-core==0.2.1",
    ],
)
