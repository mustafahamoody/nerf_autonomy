from setuptools import find_packages, setup

package_name = 'segmentation_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
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
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segmentation_node = segmentation_package.segmentation_node:main'
        ],
    },
)
