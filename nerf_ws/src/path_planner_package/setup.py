from setuptools import find_packages, setup

package_name = 'path_planner_package'

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
    maintainer='mustafa',
    maintainer_email='mustafa.a.hamoody@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_planner_node = path_planner_package.path_planner_node:main',
            'path_planner_node2 = path_planner_package.path_planner_node2:main',
            'path_planner_service = path_planner_package.path_planner_service:main',
            'path_planner_nodecm = path_planner_package.path_planner_nodecm:main',
        ],
    },
)
