import pandas as pd
import numpy as np
from enum import Enum

class DeviceState(Enum):
	OFF = 0
	ON_CAN_TX = 1
	ON_CANT_TX = 2 

INIT_OVERHEAD = 150*1e-6 # 120 uJ

# ============ helper functions ============

# TODO: implement this function in a more general way to be flexible to the energy spending policy
def sparsify_data(data_window: np.ndarray,body_parts: list,packet_size: int,leakage: float,eh,policy='opportunistic',visualize=False):
	""" Converts a 3 axis har signal into a sparse version based on energy harvested. This
		is based on an opportunistic policy (transmit when hit the threshold)

	Parameters
	----------

	data_window: np.ndarray
		A (3K+1) x T data array from K body parts each with a 3-axis accelerometer data.
		The extra column (at column 0) is the time in seconds for each sample. The data from
		Each body part is assumed to be aligned in time.
		Example columns: time | armX | armY | armZ | legX | legY | legZ | ...

	body_parts: list
		A list of strings specifying the body parts, e.g., ["arm", "leg", ...]

	packet_size: int
		number of samples in a packet

	conservative: bool
		whether to use the conservative policy

	conservative_fraction: float [1.0-2.0]
		value of relative threshold, e.g., if threshold is 0.5 and fraction is 1.1 then
		threshold becomes 0.5*1.1. After sending a packet, need to accumulate 0.5*1.1
		from wherever the energy level is at the end of transmission

	visualize: bool
		A flag to return an array of energy values for plotting

	Returns
	-------

	sparse_data: tuple
		TODO
	"""

	LEAKAGE_PER_SAMPLE = leakage*(data_window[1,0]-data_window[0,0]) # 1uW * 1/fs
	# print(LEAKAGE_PER_SAMPLE)

	# each body part is processed separately (every three channels)
	j = 1 # index of X channel for a body part
	packets = {bp: None for bp in body_parts}
	e_plots = {bp: None for bp in body_parts}


	for i,bp in enumerate(body_parts):
		# create pandas data frame as specified by EnergyHarvester.power() function
		channels = np.array([0,j,j+1,j+2]) # time + 3 acc channels of body part
		df = pd.DataFrame(data_window[:,channels],columns=['time', 'x', 'y','z'])
		j += 3 # increment to next body part
		
		# get energy as function of samples
		t_out, p_out = eh.power(df)
		e_out = eh.energy(t_out, p_out)
		valid, thresh = eh.generate_valid_mask(e_out, packet_size)

		# assume max energy we can store is 3*thresh needed to sample/TX
		MAX_E = INIT_OVERHEAD + thresh
		# print(MAX_E)

		# now we need to generate a valid mask
		valid = np.empty(len(e_out))
		valid[:] = np.nan

		# generate e_plot
		e_target = thresh.copy()
		e_plot = e_out.copy()

		if 'conservative' in policy:
			fraction = float(policy.split('_')[1])
		else:
			fraction = 1

		charge_up_thresh = fraction*thresh # conservative threshold for charging up
		if 'conservative' in policy:
			e_target = charge_up_thresh.copy()

		STATE = DeviceState.OFF

		# assume a linear energy usage over the course of a packet
		# i.e., thresh/packet_size used per sample. The array is
		# of size packet_size+1 because it starts at 0, then increments
		# by thresh/packet_size for each sample
		linear_usage = np.linspace(0,thresh,packet_size+1)
		linear_leakage = np.linspace(0,LEAKAGE_PER_SAMPLE*packet_size,packet_size+1)
		k=0
		st = np.nan
		en = np.nan
		iat_mu = np.nan
		wt = np.nan
		alpha = 0.65

		# iterate over energy values (need to change this code for other policies)
		while k < len(e_plot):
			# house keeping, make sure energy is clipped to bounds
			if e_plot[k] > MAX_E:
				e_plot[k] = MAX_E
			elif e_plot[k] < 0:
				e_plot[k] = 0

			''' ---------- Opportunistic Policy'''
			if policy == 'opportunistic':
				# print(k,k/25,STATE,e_plot[k])
				# update state
				if STATE == DeviceState.OFF: # turn on when have init overhead
					if e_plot[k] >= 5*LEAKAGE_PER_SAMPLE + INIT_OVERHEAD:
						STATE = DeviceState.ON_CANT_TX
						e_plot[k+1:] -= INIT_OVERHEAD # apply overhead instantly
				elif STATE == DeviceState.ON_CAN_TX:
					if e_plot[k] == 0: # device died
						STATE = DeviceState.OFF
					elif e_plot[k] < thresh:#+LEAKAGE_PER_SAMPLE*packet_size:
						STATE = DeviceState.ON_CANT_TX
				elif STATE == DeviceState.ON_CANT_TX:
					if e_plot[k] >= thresh:#+LEAKAGE_PER_SAMPLE*packet_size:
						STATE = DeviceState.ON_CAN_TX
					elif e_plot[k] == 0:
						STATE = DeviceState.OFF
				
				# we hit the transmit threshold
				if STATE == DeviceState.ON_CAN_TX:#(e_plot[k] > e_target):
					# we are within one packet of the end of the data
					if k + packet_size + 1 >= len(e_plot):
						valid[k:] = 1
						e_plot[k:] -= linear_usage[:len(e_plot)-k]
						e_plot[k:] -= linear_leakage[:len(e_plot)-k]
						k += (packet_size+1)
						break
					# from the index where the threshold was reached until the packet
					# has been sampled is length packet_size+1
					valid[k:k+packet_size] = 1
					e_plot[k:k+packet_size+1] -= linear_usage[:]
					e_plot[k:k+packet_size+1] -= linear_leakage[:]

					if e_plot[k+packet_size+1] > 0:
						# since e_plot is cumulative, subtract thresh from rest of it
						e_plot[k+packet_size+1:] -= thresh

					if e_plot[k+packet_size+1] > 0:
						# since e_plot is cumulative, subtract thresh from rest of it
						e_plot[k+packet_size+1:] -= LEAKAGE_PER_SAMPLE

					k += (packet_size+1)

				else:
					# print(f"{k},{k/25}")
					# print(e_plot[k:k+2])
					# apply leakage
					if e_plot[k] > 0:
						e_plot[k:] -= LEAKAGE_PER_SAMPLE
					# print(e_plot[k:k+2])
					# clip at min and max value
					if e_plot[k] > MAX_E:
						e_plot[k] = MAX_E
					elif e_plot[k] < 0:
						e_plot[k] = 0
					# print(e_plot[k:k+2])
					# print("===")
					# go to next samples
					k += 1


			elif 'conservative' in policy:
				# if bp == 'left_leg':
				#     print(k,k/25,STATE,e_plot[k]-LEAKAGE_PER_SAMPLE-e_plot[k-1],thresh,e_target)
				if e_target > MAX_E:
					e_target = MAX_E
				# update state
				if STATE == DeviceState.OFF: # turn on when have init overhead
					if e_plot[k] >= 2*LEAKAGE_PER_SAMPLE+INIT_OVERHEAD:#e_target + INIT_OVERHEAD:
						STATE = DeviceState.ON_CANT_TX
						e_plot[k+1:] -= INIT_OVERHEAD # apply overhead instantly
				elif STATE == DeviceState.ON_CAN_TX:
					if e_plot[k] == 0: # device died
						STATE = DeviceState.OFF
					elif e_plot[k] < e_target:#+LEAKAGE_PER_SAMPLE*packet_size:
						STATE = DeviceState.ON_CANT_TX
				elif STATE == DeviceState.ON_CANT_TX:
					if e_plot[k] >= e_target:#+LEAKAGE_PER_SAMPLE*packet_size:
						STATE = DeviceState.ON_CAN_TX
					elif e_plot[k] == 0:
						STATE = DeviceState.OFF

				# update state vars while device is on
				if STATE != DeviceState.OFF:
					if en is not np.nan: # increment wait time between last packet and now
						wt = k - en
				else:
					st = np.nan
					en = np.nan
					iat_mu = np.nan
					wt = np.nan
					e_target = fraction*thresh


				if STATE == DeviceState.ON_CAN_TX:#(e_plot[k] > e_target):
					# update running mean of iat
					if st is np.nan:
						st = k
					elif en is np.nan:
						en = k
						iat_mu = en-st
					else:
						st = en
						en = k
						iat_mu = alpha*(en-st)+(1-alpha)*iat_mu
						# print(f'{bp} -- time:{k/25}, st: {st}, en: {en}, iat_mu: {iat_mu}, wt: {wt}, energy: {e_plot[k]}')

					# we are within one packet of the end of the data
					if k + packet_size + 1 >= len(e_plot):
						valid[k:] = 1
						e_plot[k:] -= linear_usage[:len(e_plot)-k]
						e_plot[k:] -= linear_leakage[:len(e_plot)-k]
						k += (packet_size+1)
						break
					# from the index where the threshold was reached until the packet
					# has been sampled is length packet_size+1
					valid[k:k+packet_size] = 1
					e_plot[k:k+packet_size+1] -= linear_usage[:]
					e_plot[k:k+packet_size+1] -= linear_leakage[:]

					if e_plot[k+packet_size+1] > 0:
						# since e_plot is cumulative, subtract thresh from rest of it
						e_plot[k+packet_size+1:] -= thresh

					if e_plot[k+packet_size+1] > 0:
						# since e_plot is cumulative, subtract thresh from rest of it
						e_plot[k+packet_size+1:] -= LEAKAGE_PER_SAMPLE

					k += (packet_size+1)

					# new target
					e_target = e_plot[k]+charge_up_thresh
					if e_target > MAX_E:
						e_target = MAX_E
				
				elif STATE == DeviceState.ON_CANT_TX:
					# have enough energy and waited a while
					if e_plot[k] > thresh and wt is not np.nan and wt > 2*iat_mu:
						e_target = e_plot[k]-LEAKAGE_PER_SAMPLE # trigger a state change
						# st = np.nan
						# en = np.nan
						# iat_mu = np.nan
						# wt = np.nan
						# print(f'{bp} ** time:{k/25}, st: {st}, en: {en}, iat_mu: {iat_mu}, wt: {wt}, energy: {e_plot[k]}, e_target: {e_target}')
					# have enough energy and about to transition to cant zone
					elif e_plot[k] > thresh and ((e_plot[k] + 5*((e_plot[k]-LEAKAGE_PER_SAMPLE)-e_plot[k-1])) < thresh):
						e_target = e_plot[k]-LEAKAGE_PER_SAMPLE # trigger a state change
					#     st = np.nan
					#     en = np.nan
					#     iat_mu = np.nan
					#     wt = np.nan

					# apply leakage
					if e_plot[k] > 0:
						e_plot[k:] -= LEAKAGE_PER_SAMPLE
					# clip at min and max value
					if e_plot[k] > MAX_E:
						e_plot[k] = MAX_E
					elif e_plot[k] < 0:
						e_plot[k] = 0
					# go to next samples
					k += 1

				else:
					# apply leakage
					if e_plot[k] > 0:
						e_plot[k:] -= LEAKAGE_PER_SAMPLE
					# clip at min and max value
					if e_plot[k] > MAX_E:
						e_plot[k] = MAX_E
					elif e_plot[k] < 0:
						e_plot[k] = 0
					# go to next samples
					k += 1

			
			elif policy == 'dense':
				if STATE == DeviceState.OFF: # turn on when have init overhead
					if e_plot[k] >= 5*LEAKAGE_PER_SAMPLE + INIT_OVERHEAD:
						STATE = DeviceState.ON_CANT_TX
						e_plot[k+1:] -= INIT_OVERHEAD # apply overhead instantly
				elif STATE == DeviceState.ON_CAN_TX:
					if e_plot[k] == 0: # device died
						STATE = DeviceState.OFF
					elif e_plot[k] < thresh:#+LEAKAGE_PER_SAMPLE*packet_size:
						STATE = DeviceState.ON_CANT_TX
				elif STATE == DeviceState.ON_CANT_TX:
					if e_plot[k] >= thresh:#+LEAKAGE_PER_SAMPLE*packet_size:
						STATE = DeviceState.ON_CAN_TX
					elif e_plot[k] == 0:
						STATE = DeviceState.OFF

				# we hit the transmit threshold
				if STATE == DeviceState.ON_CAN_TX:#(e_plot[k] > e_target):
					# we are within one packet of the end of the data
					if k + packet_size + 1 >= len(e_plot):
						valid[k:] = 1
						# e_plot[k:] = min(MAX_E,e_plot[k])
						e_plot[k:] -= linear_leakage[:len(e_plot)-k]
						k += (packet_size+1)
						break
					# from the index where the threshold was reached until the packet
					# has been sampled is length packet_size+1
					valid[k:k+packet_size] = 1
					# e_plot[k:k+packet_size+1] -= linear_usage[:]
					e_plot[k:k+packet_size+1] -= linear_leakage[:]

					surp = (e_plot[k+packet_size+1] - MAX_E)


					if surp > 0:
						e_plot[k+packet_size:] -= 2*surp


					# if e_plot[k+packet_size+1] > 0:
					#     # since e_plot is cumulative, subtract thresh from rest of it
					#     e_plot[k+packet_size+1:] -= thresh
					# e_plot[k:] = min(MAX_E,e_plot[k])
					
					if e_plot[k+packet_size+1] > 0:
						# since e_plot is cumulative, subtract thresh from rest of it
						e_plot[k+packet_size+1:] -= LEAKAGE_PER_SAMPLE*packet_size


					k += (packet_size+1)

				else:
					# print(f"{k},{k/25}")
					# print(e_plot[k:k+2])
					# apply leakage
					if e_plot[k] > 0:
						e_plot[k:] -= LEAKAGE_PER_SAMPLE
					# print(e_plot[k:k+2])
					# clip at min and max value
					if e_plot[k] > MAX_E:
						e_plot[k] = MAX_E
					elif e_plot[k] < 0:
						e_plot[k] = 0
					# print(e_plot[k:k+2])
					# print("===")
					# go to next samples
					k += 1
			
		''' ----------- Package Data after applying policies -------- '''

		e_plots[bp] = e_plot
		# masking the data based on energy (this is where we have differentiability issues)
		for acc in 'xyz':
			df[acc+'_eh'] = df[acc] * valid

		# get the transition points of the masked data to see where packets start and end
		og_data = df[acc+'_eh'].values
		rolled_data = np.roll(og_data, 1)
		rolled_data[0] = np.nan # in case we end halfway through a valid packet
		nan_to_num_transition_indices = np.where(~np.isnan(og_data) & np.isnan(rolled_data))[0] # arrival idxs
		num_to_nan_transition_indices = np.where(np.isnan(og_data) & ~np.isnan(rolled_data))[0] # ending idxs
		
		# now get the actually sampled data as a list of windows
		arr = df[['x_eh','y_eh','z_eh']].values
		packet_data = [                                                                               # this zip operation is important because if we end halfway through a packet it is skipped (number of starts and ends must match)
						arr[packet_start_idx : packet_end_idx] for packet_start_idx,packet_end_idx in zip(nan_to_num_transition_indices,num_to_nan_transition_indices)
						]
		
		# get the arrival time of each packet (note that the arrival time is the end of the data)
		# TODO: num_to_nan is 1 sample after the last sample in a packet, should we do packet_end_idx-1?
		time_idxs = df['time']
		arrival_times = [ 
						time_idxs[packet_end_idx] for packet_end_idx in num_to_nan_transition_indices
						]

		# each item in the list is a packet_size x 3 array, so we just stack into one array
		if len(packet_data) > 0:
			packet_data = np.stack(packet_data)
		# we make the list into an array of packet_size x 1
		if len(arrival_times) > 0:
			arrival_times = np.stack(arrival_times)
		
		# store as a tuple
		# entry 0 is P x 1 and entry 1 is P x packet_size x 3
		packets[bp] = (arrival_times,packet_data)

	if visualize == True:
		return packets, e_plots, thresh
	else:
		return packets
