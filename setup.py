import setuptools
import shutil
import os

setuptools.setup(
    name="PyMicArrayX",
    version="0.1",
    author="Ryosuke Kojima",
    author_email="kojima.ryosuke.8e@kyoto-u.ac.jp",
    description="microphone array signal processing library",
    long_description="A python library for microphone array signal processing",
    long_description_content_type="text/markdown",
    url="https://github.com/kojima-r/PyMicArrayX",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "micarrayx-make-noise= micarrayx.tool.make_noise:main",
            "micarrayx-add-noise= micarrayx.tool.add_noise:main",
            "micarrayx-normalize= micarrayx.tool.normalize:main",
            "micarrayx-experiment= micarrayx.tool.experiment_sim00:main",
            "micarrayx-separate= micarrayx.tool.separation:main",
            "micarrayx-mix= micarrayx.tool.mix:main",
            "micarrayx-sep= micarrayx.tool.sep:main",
            "micarrayx-sim= micarrayx.simulator.sim:main",
            "micarrayx-sim-tf= micarrayx.simulator.sim_tf:main",
            "micarrayx-sim-td= micarrayx.simulator.sim_td:main",
            "micarrayx-localize= micarrayx.localization.localize:main",
            "micarrayx-localize-music= micarrayx.localization.music:main",
            "micarrayx-filter-wiener= micarrayx.filter.wiener:main",
            "micarrayx-filter-gsc= micarrayx.filter.gsc:main",
       ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
