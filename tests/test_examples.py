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

from pathlib import Path
import importlib
import pytest


example_path = Path(__file__).parent.parent / 'examples'
examples_files = [f for f in example_path.glob('*.py') if 'pyscf' not in f.name]
timings = {}


@pytest.fixture(params=examples_files, ids=lambda x: x.name)
def example_file(request):
    return request.param


@pytest.mark.timeout(300)
def test_example(example_file):
    spec = importlib.util.spec_from_file_location(example_file.name, str(example_file))
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
