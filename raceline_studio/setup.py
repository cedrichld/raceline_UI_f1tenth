from setuptools import setup

package_name = 'raceline_studio'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_data={package_name: ['templates/*.html', 'static/vendor/*']},
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='cedric',
    maintainer_email='cedrich@seas.upenn.edu',
    description='Flask-based raceline editor with live push to pure_pursuit and MPPI nodes.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'studio = raceline_studio.app:main',
        ],
    },
)
