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
import requests
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile
import tarfile
import pathlib
import os
import numpy as np
class DLProgress(tqdm):
	last_block = 0

	def hook(self, block_num=1, block_size=1, total_size=None):
		self.total = total_size
		self.update((block_num - self.last_block) * block_size)
		self.last_block = block_num

def unzip(zip_file, destination):
	if not isdir(destination):
		with zipfile.ZipFile(zip_file) as zf:
			zf.extractall(destination)

def untar(tar_file, destination):
	if not isdir(destination):
		with tarfile.open(tar_file) as tar:
			tar.extractall(path=destination)
			tar.close()
			
def getFile(url, storePath = None, cacheDir = None, subDir = 'dataset'):
	if cacheDir is None:
		cacheDir = os.path.join(os.path.expanduser('~'), '.easyai') 
	#dataDirBase = os.path.expanduser(cacheDir)
	dataDirBase = cacheDir
	try:
		if not os.path.exists(dataDirBase):
			os.makedirs(dataDirBase)
		if storePath is not None:
			dataDir =  os.path.join(dataDirBase,storePath, subDir)
		else:
			dataDir =  os.path.join(dataDirBase, subDir)
		if not os.path.exists(dataDir):
			os.makedirs(dataDir)
	except:
		if not os.access(dataDirBase, os.W_OK):
			dataDirBase = os.path.join('/tmp', '.medicalai')
		dataDir = dataDirBase
		if not os.path.exists(dataDir):
			os.makedirs(dataDir)

	file = url.split('/')[-1]
	file = os.path.join(dataDir,file)

	if not isfile(file):
		with DLProgress(unit='B', unit_scale=True, miniters=1, desc=subDir) as pbar:
			#urlretrieve(url,file,pbar.hook)
			r = requests.get(url)
			with open(file, 'wb') as f:
				f.write(r.content)

	typeF = pathlib.Path(file).suffix
	fileName = pathlib.Path(file).stem.split('.')[0]
	
	if storePath is None:
		storePath = os.path.join(dataDir,fileName)	
	if typeF == '.zip':
		unzip(file,storePath)
		return storePath
	elif typeF == '.gz':
		untar(file,storePath)
		return storePath
	else:
		return file
		
from urllib.parse import urlparse	
def check_if_url(x):
    try:
        myP = urlparse(x)
        return all([myP.scheme, myP.netloc, myP.path])
    except:
        return False

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image

def load_image(link, target_size=(32,32), storePath = None, cacheDir = None, subDir = 'images'):
	if isinstance(link, np.ndarray):
		inputI = link
	else:
		if os.path.isfile(link):
			inputI = image.load_img(link, target_size=target_size)
		elif check_if_url(link):
			inputI = image.load_img(getFile(link,storePath, cacheDir, subDir), target_size=target_size )
			
		inputI = image.img_to_array(inputI)
		inputI = np.expand_dims(inputI, axis=0) 
	return inputI