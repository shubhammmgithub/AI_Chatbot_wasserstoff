from setuptools import setup, find_packages

setup(
    name="wasserstoff_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "google-generativeai",
        "qdrant-client",
        "langchain-google-vertexai",
    ],
)