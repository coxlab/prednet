import setuptools

setuptools.setup(
    name='prednet',
    version='0.1',
    packages=setuptools.find_packages(),
    python_requires='>=3.5.*',
    install_requires=[
        'tensorflow-gpu==1.13.1',
        'Keras==2.2.4',
        'requests',
        'bs4',
        'imageio',
        'scipy==1.2.0',
        'pillow',
        'hickle'])
