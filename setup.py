from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(
   name='ConAE',
   version='0.1.0',
   description='Dimension Reduction for Efficient Dense Retrieval via Conditional Autoencoder',
   url='https://github.com/NEUIR/ConAE',
   install_requires=[
        'transformers==4.15.0',
        'pytrec-eval',
        'faiss-cpu',
        'wget',
    ],
)