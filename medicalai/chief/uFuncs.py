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



from tqdm import tqdm
def calculate_ips(ds, steps=100, batch_size=32, verbose=False):
  start = time.time()
  it = iter(ds)
  if verbose:
    for i in tqdm(range(steps)):
        batch = next(it)
        if i%10 == 0:
            print('.',end='')
  else:
    for i in range(steps):
        batch = next(it)
        if i%10 == 0:
            print('.',end='')  
  print()
  end = time.time()

  duration = end-start
  IPS = batch_size*steps/duration
  print("{} batches: {:.2f}s {:0.3f} Images/s".format(steps, duration, IPS))
  return IPS, duration