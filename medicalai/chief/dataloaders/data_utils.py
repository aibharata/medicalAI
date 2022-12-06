#    Copyright 2020-2022 AIBharata Emerging Technologies Pvt. Ltd.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import absolute_import

def safe_labelmap_converter(labelMap):
	labs = [0 for x in list(labelMap.keys())]
	for k,v in labelMap.items():
		labs[v]=k
	return labs

def safe_label_to_labelmap_converter(labels):
	labelMap = {}
	for x in range(0,len(labels)):
		labelMap[str(labels[x])]=x
	return labelMap

class myDict(dict):
    pass