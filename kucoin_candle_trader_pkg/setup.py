from setuptools import setup, find_packages

setup(
    name='kucoin_candle_trader',
    version='0.1.0',
    description='A library for live data updates and fetching spot data from KuCoin',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Andre',
    author_email='receivemailforme@gmail.com',
    url='https://github.com/gigishub/kucoin_candle_trader',  # Update with your repository URL
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0.0',
        'requests>=2.0.0',
        'websocket-client>=0.57.0',
        'dataclasses; python_version<"3.7"',  # Only needed for Python versions < 3.7
        'typing-extensions; python_version<"3.8"',  # Only needed for Python versions < 3.8
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)