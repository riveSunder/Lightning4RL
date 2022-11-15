from setuptools import setup

setup(name="Lightning4RL",
    packages=["l4rl"],\
    version="0.0",\
    install_requires=["lightning==1.8.1",\
        "torch==1.9.0",\
        "pyglet==1.5.27",\
        "procgen==0.10.7",\
        "gym==0.20.0"])
