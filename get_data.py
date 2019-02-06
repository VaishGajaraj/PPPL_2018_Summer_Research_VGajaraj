import matplotlib.pyplot as plt
import numpy as np
import pandas
import h5py as h5
#import MDSplus as mds


NUM_GRID_POINTS = 20 #the number of spacial readings


def profile_column_names(profile,n):
	'''Create a list of strings, some word followed by numbers 1 to n'''
	return [profile+'{:02d}'.format(i+1) for i in np.arange(n)]

def run_to_num(run):
	'''Convert TRANSP run, e.g., A01, to numerical equivalent, e.g., 0101'''
	return '{:02d}'.format(ord(run[0].lower())-96)+run[1:]

def form_run_id(shot,run):
	'''Convert TRANSP shot and run, e.g., 204118  and A01, to numerical equivalent, e.g., 2041180101'''
	return int(str(shot)+run_to_num(run))

def get_signal(connection, tree, shot, tag):
	# get 1D signal from MDSplus tree
	connection.openTree(tree, shot)
	data = connection.get(tag).value
	time = connection.get('dim_of(' + tag + ')').value
	connection.closeAllTrees()
	return time, data
	
def get_run_data(shot, run, scalar_variables, profile_variables, connection, tree):
	run_id = form_run_id(shot,run)
	scalar_data = []
	for scalar in scalar_variables:
		#dn0out isn't in output so its a special case
		if scalar.upper() == 'DN0OUT':
			#print('working on dn0out')
			t, namelist = get_signal(connection, tree, run_id, '.NAME_LIST')
			t, bpshi = get_signal(connection, tree, run_id, '.TRANSP_OUT:BPSHI') #to get time
			try:
				dn0out = float([s for s in namelist if 'DN0OUT' in s.upper()][0].split('=')[1].split('!')[0])
				scalar_data.append(dn0out*np.ones(t.shape[0]))
			except:
				dn0out = 1.0e12 #default value if not set in namelist
				scalar_data.append(dn0out * np.ones(t.shape[0]))
		elif scalar.upper() == 'R0': #major radius [m]
			x, rmjmp = get_signal(connection, tree, run_id, '.TRANSP_OUT:RMJMP')
			scalar_data.append(rmjmp[:,-1]/100.0)
		elif scalar.upper() == 'AMIN': #minor radius [m]
			x, rmnmp = get_signal(connection, tree, run_id, '.TRANSP_OUT:RMNMP')
			scalar_data.append(rmnmp[:,-1]/100.0)
		elif scalar.upper() == 'TRIANGU': #upper triangularity
			x, triangu = get_signal(connection, tree, run_id, '.TRANSP_OUT:TRIANGU')
			scalar_data.append(triangu[:,-1])
		elif scalar.upper() == 'TRIANGL': #lower triangularity
			x, triangl = get_signal(connection, tree, run_id, '.TRANSP_OUT:TRIANGL')
			scalar_data.append(triangl[:,-1])
		elif scalar.upper() == 'ELONG':  # elongation
			x, elong = get_signal(connection, tree, run_id, '.TRANSP_OUT:ELONG')
			scalar_data.append(elong[:, -1])
		else:
			print(scalar)
			t,data = get_signal(connection, tree, run_id, '.TRANSP_OUT:'+scalar)
			scalar_data.append(data)
		print(scalar_data[-1].shape)

	scalar_df = pandas.DataFrame.from_items(zip(['Shot','ID','Time']+scalar_variables,[shot*np.ones(t.shape[0]),int(run_to_num(run))*np.ones(t.shape[0]),t]+scalar_data))

	profile_data = []
	for profile in profile_variables:
		if profile == 'PFI':
			x, ptowb = get_signal(connection, tree, run_id, '.TRANSP_OUT:PTOWB')
			x, pplas = get_signal(connection, tree, run_id, '.TRANSP_OUT:PPLAS')
			data = ptowb-pplas
		else:
			x,data = get_signal(connection, tree, run_id, '.TRANSP_OUT:'+profile)
		from scipy.interpolate import interp1d
		x20 = np.linspace(0,1,NUM_GRID_POINTS)
		print(x.shape)
		print(x20.shape)
		print(data.shape)
		data_int = interp1d(x,data,fill_value='extrapolate',axis=1)
		df_prof = pandas.DataFrame(data_int(x20),columns=profile_column_names(profile,NUM_GRID_POINTS))
		profile_data.append(df_prof)


	profile_df = pandas.concat([scalar_df[['Shot','ID','Time']]]+profile_data,axis=1)
	return scalar_df, profile_df

def get_beast_data_for_range(shots):
	scalar_dfs = []
	profile_dfs = []

	for shot in shots:
		try:
			scalar_df, profile_df = get_run_data(shot,'X99')
			scalar_dfs.append(scalar_df)
			profile_dfs.append(profile_df)
			print("Got data for shot: " + str(shot))
		except:
			print("Failed to get data for shot: "+str(shot))
	scalar_df_out = pandas.concat(scalar_dfs,axis=0)
	profile_df_out = pandas.concat(profile_dfs,axis=0)
	return scalar_df_out, profile_df_out

def find_submitted_runs(runlist_path):
	df = pandas.read_csv(runlist_path,dtype=str)
	submitted_runs = df[np.logical_or(df.submitted == 1, df.submitted == '1')]
	return submitted_runs


def run_exists(connection, tree, shotrun):
	try:
		get_signal(connection, tree, shotrun, '.TRANSP_OUT:DIFB')
		return 1
	except:
		return 0

def get_data_for_runlist(runlist_path,connection_path='transpgrid.pppl.gov',tree='transp_nstu'):
	scalar_dfs = []
	profile_dfs = []

	connection = mds.Connection(connection_path)
	submitted = find_submitted_runs(runlist_path)
	n_runs = len(submitted)
	for i in np.arange(n_runs):
		try:
			shot = int(submitted.iloc[i]['Shot'])
			run = submitted.iloc[i]['ID']
			scalar_df, profile_df = get_run_data(shot,run,connection,tree)
			scalar_dfs.append(scalar_df)
			profile_dfs.append(profile_df)
			print("Got data for run: " + str(shot)+str(run))
		except:
			print("Failed to get data for run: "+str(shot)+str(run))
	scalar_df_out = pandas.concat(scalar_dfs,axis=0)
	profile_df_out = pandas.concat(profile_dfs,axis=0)
	return scalar_df_out, profile_df_out

def get_data_for_batchlist(batchlist,scalar_variables,profile_variables,runlist_path,connection_path='transpgrid.pppl.gov',tree='transp_nstu'):
	scalar_dfs = []
	profile_dfs = []

	connection = mds.Connection(connection_path)
	n_runs = len(batchlist)
	for i in np.arange(n_runs):
		try:
			shot = int(batchlist.iloc[i]['Shot'])
			run = batchlist.iloc[i]['ID']
			scalar_df, profile_df = get_run_data(shot,run,scalar_variables,profile_variables,connection,tree)
			scalar_dfs.append(scalar_df)
			profile_dfs.append(profile_df)
			print("Got data for run: " + str(shot)+str(run))
		except:
			print("Failed to get data for run: "+str(shot)+str(run))
	scalar_df_out = pandas.concat(scalar_dfs,axis=0)
	profile_df_out = pandas.concat(profile_dfs,axis=0)
	attrs = {'runlist':runlist_path,'tree':tree,'scalar_variables':scalar_variables,'profile_variables':profile_variables}
	batch = {'scalar_df':scalar_df_out, 'profile_df':profile_df_out,'attrs':attrs,'batchlist':batchlist}
	return batch

#TODO: save with split and prevent training on testing or validation data in other scripts
def save_batch(batch,hdf5_name):
	print('Saving data to: ' +hdf5_name)
	import os
	directory = os.path.dirname(hdf5_name)
	if not os.path.exists(directory):
		os.makedirs(directory)
	batch['profile_df'].to_hdf(hdf5_name,'profile_df')
	batch['scalar_df'].to_hdf(hdf5_name,'scalar_df')
	batch['batchlist'].to_hdf(hdf5_name,'batchlist')
	with h5.File(hdf5_name, 'a') as h5f:
		group = h5f.require_group('batch_info')
		if batch['attrs'] is not None:
			for attr in batch['attrs'].keys():
				group.attrs[attr] = batch['attrs'][attr]

def load_batch(hdf5_name):
	batch = {}
	batch['profile_df'] = pandas.read_hdf(hdf5_name,'profile_df')
	batch['scalar_df'] = pandas.read_hdf(hdf5_name,'scalar_df')
	batch['batchlist'] = pandas.read_hdf(hdf5_name,'batchlist')
	with h5.File(hdf5_name, 'a') as h5f:
		batch['attrs'] = dict(h5f['batch_info'].attrs)
	return batch

def get_batchlist_from_runlist(runlist_path, batch_size, excluded_runs=None, excluded_shots=None):
	submitted = find_submitted_runs(runlist_path)
	if excluded_runs is not None:
		df_all = submitted.merge(excluded_runs.drop_duplicates(),
							 how='left', indicator='exists')
		submitted_included = df_all[df_all['exists'] == 'left_only'].drop('exists',axis=1)
	else:
		submitted_included = submitted
	if excluded_shots is not None:
		submitted_included = submitted_included[~submitted_included['Shot'].isin(excluded_shots)]

	if len(submitted_included)>=batch_size:
		batchlist = submitted_included.iloc[np.random.choice(len(submitted_included), batch_size, replace=False)]
	else:
		print('Batch size exceeds included runs in runlist.')
		batchlist = submitted_included
	return batchlist

def concat_batches(batches):
	scalar_dfs = []
	profile_dfs = []
	batchlists = []
	for batch in batches:
		scalar_dfs.append(batch['scalar_df'])
		profile_dfs.append(batch['profile_df'])
		batchlists.append(batch['batchlist'])
	batch['scalar_df'] = pandas.concat(scalar_dfs, axis=0)
	batch['profile_df'] = pandas.concat(profile_dfs, axis=0)
	batch['batchlist'] = pandas.concat(batchlists, axis=0)
	return batch

def load_batch_data(dataset, net=None, batches_to_load=[('training',1)], trim_last_percent_of_shot=True,
					percent_to_trim=5, apply_pinj_filtering=False):
	batches = []

	for split, batch_number in batches_to_load:
		hdf5_name = './{}/{}/batch_{:02d}.h5'.format(dataset.batch_output_path, split, batch_number)
		batch = load_batch(hdf5_name)
		if trim_last_percent_of_shot:
			scalar_df_trimmed = trim_last_percent_of_shot_fnc(batch['scalar_df'], percent_to_trim)
			profile_df_trimmed = trim_last_percent_of_shot_fnc(batch['profile_df'], percent_to_trim)
			batch['scalar_df'] = scalar_df_trimmed.reset_index(drop=True)
			batch['profile_df'] = profile_df_trimmed.reset_index(drop=True)

		if apply_pinj_filtering and (net is not None):
			batch['scalar_df'] = add_filtered_pinj_to_df(batch['scalar_df'], net.X_scalars, net.pinj_taus).reset_index(drop=True)

		batches.append(batch)

	return concat_batches(batches)


def trim_last_percent_of_shot_fnc(data_df, percent):
	shots = data_df['Shot'].unique()
	for shot in shots:
		tmax = ((100.0 - percent) / 100.0) * max(data_df[data_df['Shot'] == shot]['Time'])
		data_df = data_df[np.logical_or(data_df['Shot'] != shot, data_df['Time'] < tmax)]
	return data_df


def add_filtered_pinj_to_df(data_df, X_scalars, pinj_taus):
	lpf_dfs = []
	for x in [x for x in X_scalars if x.startswith('pinj')]:
		col = x
		pinj_lpf = np.zeros((len(data_df[col]), len(pinj_taus)))
		pinj_lpf[0, :] = data_df.iloc[0][col]
		for i in np.arange(pinj_lpf.shape[0] - 1):
			if (data_df['ID'].iloc[i + 1] == data_df['ID'].iloc[i]) and (
						data_df['Shot'].iloc[i + 1] == data_df['Shot'].iloc[i]):
				pinj_lpf[i + 1, :] = pinj_lpf[i, :] + (data_df['Time'].iloc[i + 1] - data_df['Time'].iloc[i]) * (
					-pinj_lpf[i] / np.array(pinj_taus) + data_df[col].iloc[i] / np.array(pinj_taus))

		lpf_dfs.append(
			pandas.DataFrame(pinj_lpf, columns=[x + '_lpf_' + str(j + 1) for j in np.arange(len(pinj_taus))]))
	lpf_df = pandas.concat(lpf_dfs, axis=1)
	return pandas.concat([data_df, lpf_df], axis=1)

class dataset:
	def __init__(self,batch_output_path,scalar_variables,profile_variables,testing_fraction=0.1,validation_fraction=0.1):
		self.scalar_variables = scalar_variables
		self.profile_variables = profile_variables
		self.testing_fraction = testing_fraction
		self.validation_fraction = validation_fraction
		self.runlists = []
		self.n_batches = {'all':0,'training':0,'testing':0,'validation':0}
		self.shots = {'all':[],'training':[],'testing':[],'validation':[]}
		self.runs_read = {'all':None,'training':None,'testing':None,'validation':None}
		self.training_shots = []
		self.testing_shots = []
		self.validation_shots = []
		self.batch_output_path = batch_output_path

	def add_runlist(self,runlist_path):
		if runlist_path not in self.runlists:
			self.runlists.append(runlist_path)
		submitted_runs = find_submitted_runs(runlist_path)
		new_shots = submitted_runs[~submitted_runs['Shot'].isin(self.shots['all'])]['Shot'].unique()
		self.assign_shots(new_shots)

	def assign_shots(self, new_shots):
		self.shots['all'].extend(new_shots)
		assign_to_training = np.random.binomial(1, 1-self.testing_fraction-self.validation_fraction, len(new_shots))
		self.shots['training'].extend(new_shots[assign_to_training==1])
		new_t_and_v_shots = new_shots[assign_to_training==0]
		assign_to_testing = np.random.binomial(1,self.testing_fraction/(self.testing_fraction+self.validation_fraction), len(new_t_and_v_shots))
		self.shots['testing'].extend(new_t_and_v_shots[assign_to_testing==1])
		self.shots['validation'].extend(new_t_and_v_shots[assign_to_testing==0])

	def generate_batches(self,runlist_path,split='training',n_batches=1,batch_size=10,connection_path='transpgrid.pppl.gov'):
		if n_batches==0:
			submitted_runs = find_submitted_runs(runlist_path)
			n_submitted = len(submitted_runs)
			n_runs_read = 0
			if self.runs_read['all'] is not None:
				n_runs_read = len(self.runs_read['all'])
			n_batches = np.floor_divide((n_submitted-n_runs_read),batch_size)
		for i in np.arange(n_batches):
			excluded_shots = np.setdiff1d(self.shots['all'],self.shots[split])
			batchlist = get_batchlist_from_runlist(runlist_path,batch_size=batch_size,excluded_runs=self.runs_read[split],excluded_shots=excluded_shots)
			if len(batchlist) > 0:
				if self.runs_read[split] is None:
					self.runs_read[split] = batchlist
				else:
					self.runs_read[split] = pandas.concat([self.runs_read[split], batchlist], axis=0)
				if self.runs_read['all'] is None:
					self.runs_read['all'] = batchlist
				else:
					self.runs_read['all'] = pandas.concat([self.runs_read['all'], batchlist], axis=0)

				batch_data = get_data_for_batchlist(batchlist, self.scalar_variables, self.profile_variables, runlist_path,
													connection_path=connection_path, tree=tree)
				self.n_batches[split]+=1
				self.n_batches['all']+=1
				# suffix = 'runlist_test2wdn0out'
				hdf5_name = './'+self.batch_output_path + '/'+split + '/batch_'+ str(self.n_batches[split]).zfill(2)+'.h5'
				#
				save_batch(batch_data,hdf5_name)
				self.save()


	def save(self):
		#todo: fix using http://stefaanlippens.net/python-pickling-and-dealing-with-attributeerror-module-object-has-no-attribute-thing.html
		import pickle
		import os
		file_path = './'+self.batch_output_path + '/dataset.p'
		directory = os.path.dirname(file_path)
		if not os.path.exists(directory):
			os.makedirs(directory)
		pickle.dump(self, open(file_path, "wb"))

if __name__ == '__main__':

	nbeams = 6
	scalar_variables = ['BPSHI', 'DN0OUT', 'NEUTT', 'PCURC', 'ELONG','R0','BZXR','AMIN','PVOL','TRIANGU','TRIANGL','BPCX0', 'BPCXX', 'BPLIM', 'ZEFFC']+['pinj0' + str(i) for i in np.arange(nbeams)+1]+['einj0' + str(i) + '_e1' for i in np.arange(nbeams)+1]
	profile_variables = ['TE', 'NE', 'CURB', 'Q', 'PBE', 'PBI', 'DIFB', 'TQBE', 'TQBI', 'PFI', 'CURBS', 'BDENS','ETA_SNC']
	#scalar_df,profile_df = get_run_data(204118,'X99')
	#runlist_path = 'runlist_040218.txt'
	runlist_path = 'runlist_hifi.txt'
	connection_path = 'transpgrid.pppl.gov'
	tree = 'transp_nstu'

	ds = dataset('datasets/magdif',scalar_variables=scalar_variables,profile_variables=profile_variables)
	dataset.__module__ = 'get_data'
	ds.add_runlist(runlist_path)
	ds.generate_batches(runlist_path, split='testing', n_batches=5, batch_size=20, connection_path=connection_path)
	ds.generate_batches(runlist_path, split='validation', n_batches=5, batch_size=20, connection_path=connection_path)
	ds.generate_batches(runlist_path, split='training', n_batches=40, batch_size=20, connection_path=connection_path)
	ds.save()
