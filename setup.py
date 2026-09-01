from setuptools import setup, find_packages

setup(
    name="loan-performance-intelligence-engine",
    version="1.0.0",
    description="ML system for loan-data profiling, performance prediction, anomaly detection, scenario simulation, explainability, and LLM-assisted review.",
    author="Intain Campus FinTech Challenge 2026",
    python_requires=">=3.9",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.2",
        "lightgbm>=4.0",
        "lifelines>=0.27",
        "shap>=0.42",
        "matplotlib>=3.6",
        "seaborn>=0.12",
        "scipy>=1.10",
        "pyyaml>=6.0",
        "httpx>=0.24",
        "tenacity>=8.2",
        "joblib>=1.2",
        "click>=8.1",
        "jupyter>=1.0",
        "ipykernel>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "ruff", "mypy"],
        "mlflow": ["mlflow>=2.0"],
    },
    entry_points={
        "console_scripts": [
            "loan-engine=run_pipeline:run",
        ]
    },
)
