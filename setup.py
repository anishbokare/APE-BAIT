from setuptools import setup, find_packages

setup(
    name="ape-bait",
    version="1.0.0",
    description="Adversarial Perturbation Engine to Blind Attacker AI Tools",
    author="Security Research",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scapy>=2.5.0",
        "flask>=3.0.0",
        "flask-socketio>=5.3.6",
        "pyyaml>=6.0.1",
        "click>=8.1.7",
        "rich>=13.7.0",
        "tqdm>=4.66.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "ape-bait=src.core.engine:cli",
            "ape-bait-dashboard=src.dashboard.app:cli",
            "ape-bait-validate=scripts.run_validation:cli",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "Programming Language :: Python :: 3.10",
    ],
)
