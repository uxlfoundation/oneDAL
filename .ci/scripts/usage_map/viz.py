#===============================================================================
# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

"""viz.py <build> -> out-<build>/usage-map.html (single file, data inlined)"""
import json
import os
import sys

U = os.environ.get("USAGE_MAP_WORK", os.getcwd())
B = sys.argv[1]
data = json.load(open(f"{U}/out-{B}/data.json"))
html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz-template.html")).read().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
open(f"{U}/out-{B}/usage-map.html", "w").write(html)
print(f"{U}/out-{B}/usage-map.html", len(html) // 1024, "KB")
