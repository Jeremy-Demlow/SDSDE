from pkg_resources import parse_version
from configparser import ConfigParser
import setuptools
assert parse_version(setuptools.__version__) >= parse_version('36.2')

# note: all settings are in settings.ini; edit there, not here
config = ConfigParser(delimiters=['='])
config.read('settings.ini')
cfg = config['DEFAULT']

cfg_keys = 'version description keywords author author_email'.split()
# cfg_keys = 'description keywords author author_email'.split()
expected = cfg_keys + \
    "lib_name user branch license status min_python audience language".split()
for o in expected:
    assert o in cfg, "missing expected setting: {}".format(o)
setup_cfg = {o: cfg[o] for o in cfg_keys}

licenses = {
    'Connectresorts': ('Jeremy Demlow License 1.0', 'None'),
}
statuses = ['1 - Planning', '2 - Pre-Alpha', '3 - Alpha',
            '4 - Beta', '5 - Production/Stable', '6 - Mature', '7 - Inactive']
py_versions = '2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8'.split()

requirements = ['pip',
                'packaging',
                'snowflake-connector-python==2.6.2',
                'snowflake-sqlalchemy==1.3.2',
                'ruamel.yaml',
                'pysftp',
                'pyarrow==5.0.0',
                'fastcore',
                'xgboost',
                'scikit-learn',
                'hyperopt==0.2.7',
                'scikit-plot',
                'rfpimp',
                'azure-storage-file-datalake==12.4.0',
                'azure-storage-blob==12.8.1',
                'azure-core==1.13.0'
                ]
# lic = licenses[cfg['license']]
min_python = cfg['min_python']

setuptools.setup(
    name=cfg['lib_name'],
    classifiers=[
        'Development Status :: ' + statuses[int(cfg['status'])],
        'Intended Audience :: ' + cfg['audience'].title(),
        # 'License :: ' + lic[1],
        'Natural Language :: ' + cfg['language'].title(),
    ] + ['Programming Language :: Python :: '+o for o in py_versions[py_versions.index(min_python):]],
    url=cfg['git_url'],
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires='>=' + cfg['min_python'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    zip_safe=False,
    entry_points={'console_scripts': cfg.get('console_scripts', '').split()},
    **setup_cfg)
