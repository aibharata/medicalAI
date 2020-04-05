import time
def timeit(func):
	def ntimes(*args, **kw):
		start = time.time()
		value = func(*args, **kw)
		end = time.time()
		if 'log_time' in kw:
			name = kw.get('log_name', func.__name__.upper())
			kw['log_time'][name] = int((end - start) * 1000)
		else:
			print('%r  %2.2f ms' % \
				  (func.__name__, (end - start) * 1000))
		return value
	return ntimes