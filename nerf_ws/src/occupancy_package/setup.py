from setuptools import find_packages, setup
from glob import glob

package_name = 'occupancy_package'

setup(
    name=package_name,
    version='0.0.0',
    # packages=find_packages(exclude=['test']),
    packages=find_packages(include=['occupancy_package', 'occupancy_package.*',]),
    include_package_data=True,

    package_data = {'nerf_config.libs.raymarching': ['src/*.cu', 'src/*.cpp', 'src/*.h'], 
                    'nerf_config.libs.gridencoder': ['src/*.cu', 'src/*.cpp', 'src/*.h'], 
                    'nerf_config.libs.gridencoder.freqencoder': ['src/*.cu', 'src/*.cpp', 'src/*.h'],
                    'nerf_config.libs.shencoder': ['src/*.cu', 'src/*.cpp', 'src/*.h'],
                    'occupancy_package.config': ['*.yaml', '*.yml'],
                    'occupancy_package.model_weights.stone_nerf.checkpoints': ['*.pth'],},

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kishorey',
    maintainer_email='kishore.yogaraj@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require = {'test':['pytest'],},
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'occupancy_node = occupancy_package.node:main'
        ],
    },
)
