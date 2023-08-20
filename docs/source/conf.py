#     Copyright 2023 Max Nusspickel
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import inspect
from operator import attrgetter

sys.path.insert(1, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'BTensor'
copyright = '2023, Max Nusspickel'
author = 'Max Nusspickel'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'sphinx.ext.linkcode',
              ]

templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']

# Autodoc

autodoc_default_options = {
    'undoc-members': True,
}

autodoc_member_order = 'groupwise'
autodoc_typehints = 'description'

# Autosummary

autosummary_generate = True


# Linkcode

def linkcode_resolve(domain, info):
    package = 'btensor'
    if domain not in ("py", "pyx"):
        return
    if not info.get("module") or not info.get("fullname"):
        return

    class_name = info["fullname"].split(".")[0]
    module = __import__(info["module"], fromlist=[class_name])
    obj = attrgetter(info["fullname"])(module)

    # Unwrap the object to get the correct source
    # file in case that is wrapped by a decorator
    obj = inspect.unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return

    fn = os.path.relpath(fn, start=os.path.dirname(__import__(package).__file__))
    url_fmt = 'https://github.com/maxnus/btensor/blob/main/src/{package}/{path}'
    try:
        lineno = inspect.getsourcelines(obj)[1]
        url_fmt += '#L{lineno}'
        return url_fmt.format(package=package, path=fn, lineno=lineno)
    except OSError:
        return url_fmt.format(package=package, path=fn)
