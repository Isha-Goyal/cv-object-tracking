from setuptools import find_packages, setup

package_name = 'cv_object_tracking'

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
    maintainer='igoyal',
    maintainer_email='goyalisha03@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
'ball_tracker = cv_object_tracking.ball_tracker:main',
'meanshift = cv_object_tracking.meanshift:main',
'sift = cv_object_tracking.sift:main'
        ],
    },
)
