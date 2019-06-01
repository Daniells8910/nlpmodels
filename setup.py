# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import sys
if sys.version_info.major < 3:
    sys.setdefaultencoding('utf-8')
    import codecs
    open=codecs.open

from setuptools import setup, find_packages
module_name = 'nlpmodels'
version = '0.0.1'

if __name__ == '__main__':
    import sys
    print(sys.argv)
    sys.argv.append('bdist_wheel')

    setup(
        name=module_name,
        version=version,
        url='',
        long_description=open('README.md', encoding='utf-8').read(),
        packages=find_packages(exclude=[]),
        include_package_data=True,
        install_requires=[
            'jieba',
            'codecs',
        ],
        extras_require={'torch': ['torch'],
                        'torchvision': ['torchvision']},
    )
#打包wheel命令:
# python setup.py bdist_wheel