import numpy as np
from sarewt.data_reader import CaseDataReader
from vande.util.data_generator import CaseDataGenerator
import pandas as pd
import h5py

def events_to_input_samples(constituents):
    '''Stacks up both jets in a constituents array of shape N x 2 x 100 x 3 and returns an array of shape 2N x 100 x 3'''
    const_j1 = constituents[:,0,:,:]
    const_j2 = constituents[:,1,:,:]
    return np.vstack([const_j1, const_j2])

def events_to_orig_reco_samples(constituents_orig,constituents_reco):
    '''
        Stacks up both jets in a constituents array of shape N x 2 x 100 x 3 and returns an array of shape 2N x 100 x 3
        Does this for two sets of jets: orig and reco
    '''
    
    orig_const_j1 = constituents_orig[:,0,:,:]
    orig_const_j2 = constituents_orig[:,1,:,:]
    reco_const_j1 = constituents_reco[:,0,:,:]
    reco_const_j2 = constituents_reco[:,1,:,:]
    return np.vstack([orig_const_j1, orig_const_j2]),np.vstack([reco_const_j1, reco_const_j2])

class CMSDataGenerator(CaseDataGenerator):
    ''' 
    We can inherit from the original Case Data Reader and modify only the part where pt,eta,phi is converted to px,py,pz
    For the normalizing flow, we don't need to convert 
    '''
    def __init__(self, path, sample_part_n=1e4, sample_max_n=None):
        super().__init__(path, sample_part_n, sample_max_n)
        self.path=path
        print("Successfully initialized")
    def __call__(self,orig_or_reco='orig'): # -> generator object yielding np.ndarray, np.ndarray
        '''
            generate single(!) data-sample (batching done in tf.Dataset)
        '''
        
        # create new file data-reader, each time data-generator is called (otherwise file-data-reader generation not reset)
        generator = CMSDataHandler(self.path).generate_event_parts_from_dir(parts_n=self.sample_part_n)
        
        samples_read_n = 0
        # loop through whole dataset, reading sample_part_n events at a time
        for constituents_orig, constituents_reco in generator:
            
            orig_samples,reco_samples = events_to_orig_reco_samples(constituents_orig[:,:,:,:3],constituents_reco[:,:,:,:3])
            #import pdb;pdb.set_trace()
            indices = list(range(len(reco_samples)))
            samples_read_n += len(reco_samples)
            while indices:
                index = indices.pop(0)
                next_reco_sample = reco_samples[index] #.copy() 
                next_orig_sample = orig_samples[index] #.copy() 
                yield next_orig_sample,next_reco_sample
            if self.sample_max_n is not None and (samples_read_n >= self.sample_max_n):
                break
        print('[DataGenerator]: __call__() yielded {} samples'.format(samples_read_n))
        generator.close()

class CMSDataHandler(CaseDataReader):
    def __init__(self,path):
        super().__init__(path)
        self.jet_orig_constituents_key='jetOrigConstituentsList'
        self.jet_reco_constituents_key='jetConstituentsList'    
        

    def read_events_from_file(self, fname=None): # -> np.ndarray, np.ndarray
        fname = fname or self.path
        constituents_orig, constituents_reco = self.read_orig_and_reco_constituents_from_file(fname) # -> np.ndarray, np.ndarray
        try:
            constituents_orig, constituents_reco = self.read_orig_and_reco_constituents_from_file(fname) # -> np.ndarray, np.ndarray
        except OSError as e:
            print("\n[ERROR] Could not read file ", fname, ': ', repr(e))
        except IndexError as e:
            print("\n[ERROR] No data in file ", fname, ':', repr(e))
        except Exception as e:
            print("\nCould not read file ", fname, ': ', repr(e))
        return np.asarray(constituents_orig), np.asarray(constituents_reco)
    
    def read_events_from_dir(self, read_n=None, reco_or_orig='both', **cuts): # -> np.ndarray, list, np.ndarray, list
        '''
        read dijet events (jet constituents & jet features) from files in directory
        :param read_n: limit number of events
        :return: concatenated jet constituents and jet feature array + corresponding particle feature names and event feature names
        '''
        print('[DataReader] read_events_from_dir(): reading {} events from {}'.format((read_n or 'all'), self.path))

        constituents_orig_concat = []
        constituents_reco_concat = []

        flist = self.get_file_list()
        n = 0
        
        for i_file, fname in enumerate(flist):
            constituents_orig, constituents_reco = self.read_events_from_file(fname)
            constituents_orig_concat.append(constituents_orig)
            constituents_reco_concat.append(constituents_reco)
            n += len(constituents_orig)
            if read_n is not None and (n >= read_n):
                break
        
        constituents_orig_concat, constituents_reco_concat = np.concatenate(constituents_orig_concat, axis=0)[:read_n], np.concatenate(constituents_reco_concat, axis=0)[:read_n]
        print('\nnum files read in dir ', self.path, ': ', i_file + 1)
        if reco_or_orig=='reco':
            return np.asarray(constituents_reco_concat)
        elif reco_or_orig=='orig':
            return np.asarray(constituents_orig_concat)
        else:
            return [np.asarray(constituents_orig_concat), np.asarray(constituents_reco_concat)]

    def extend_by_file_content(self, constituents_orig, constituents_reco, fname):
        cco, ccr = self.read_events_from_file(fname)
        constituents_orig.extend(cco)
        constituents_reco.extend(ccr)
        return constituents_orig, constituents_reco

    def generate_event_parts_by_num(self, parts_n, flist):
        # keeping data in lists for performance
        constituents_orig_concat = []
        constituents_reco_concat = []

        for i_file, fname in enumerate(flist):
            constituents_orig_concat, constituents_reco_concat = self.extend_by_file_content(constituents_orig_concat, constituents_reco_concat, fname)

            while (len(constituents_reco_concat) >= parts_n): # if event sample size exceeding max size or min n, yield next chunk and reset
                constituents_orig_part, constituents_orig_concat = constituents_orig_concat[:parts_n], constituents_orig_concat[parts_n:] # makes copy of *references* to ndarrays 
                constituents_reco_part, constituents_reco_concat = constituents_reco_concat[:parts_n], constituents_reco_concat[parts_n:] # makes copy of *references* to ndarrays 
                yield (np.asarray(constituents_orig_part), np.asarray(constituents_reco_part)) # makes copy of all data, s.t. yielded chunk is new(!) array (since *_part is a list) => TODO: CHECK!
        
        # if data left, yield it
        if constituents_reco_concat:
            yield (np.asarray(constituents_orig_concat), np.asarray(constituents_reco_concat))


    def generate_event_parts_from_dir(self, parts_n=None, parts_sz_mb=None):
        '''
        file parts generator
        yields events in parts_n (number of events) or parts_sz_mb (size of events) chunks
        '''
        
        # if no chunk size or chunk number given, return all events in all files of directory
        
        flist = self.get_file_list()

        if parts_n is not None:
            gen = self.generate_event_parts_by_num(int(parts_n), flist)
        else: 
            gen = self.generate_event_parts_by_size(parts_sz_mb, flist)

        for chunk in gen: 
            yield chunk


    def read_jet_constituents_from_file(self, file):
        ''' return jet constituents as array of shape N x 2 x 100 x 3
            (N examples, each with 2 jets, each jet with 100 highest-pt particles, each particle with pt, eta, phi features)
        '''
        #print(file)
        if isinstance(file, str):
            file = h5py.File(file,'r') 
        j1_constituents_pt_eta_phi = np.array(file.get(self.jet_orig_constituents_key))[0,:,:,:] # (batch, 100, 3)
        j2_constituents_pt_eta_phi = np.array(file.get(self.jet_orig_constituents_key))[1,:,:,:] # (batch, 100, 3)
        
        j1_reco_constituents_pt_eta_phi = np.array(file.get(self.jet_reco_constituents_key))[0,:,:,:] # (batch, 100, 3)
        j2_reco_constituents_pt_eta_phi = np.array(file.get(self.jet_reco_constituents_key))[1,:,:,:] # (batch, 100, 3)
        
        # x_j1 = np.argsort(np.asarray(j1_constituents_pt_eta_phi)[...,0]*(-1), axis=1)
        # j1_constituents_pt_eta_phi_sorted = np.take_along_axis(np.asarray(j1_constituents_pt_eta_phi), x_j1[...,None], axis=1)
        # x_j2 = np.argsort(np.asarray(j2_constituents_pt_eta_phi)[...,0]*(-1), axis=1)
        # j2_constituents_pt_eta_phi_sorted = np.take_along_axis(np.asarray(j2_constituents_pt_eta_phi), x_j2[...,None], axis=1)

        # j1_constituents = np.array(j1_constituents_pt_eta_phi_sorted)
        # j2_constituents = np.array(j2_constituents_pt_eta_phi_sorted)
        
        # x_reco_j1 = np.argsort(np.asarray(j1_reco_constituents_pt_eta_phi)[...,0]*(-1), axis=1)
        # j1_reco_constituents_pt_eta_phi_sorted = np.take_along_axis(np.asarray(j1_reco_constituents_pt_eta_phi), x_j1[...,None], axis=1)
        # x_reco_j2 = np.argsort(np.asarray(j2_reco_constituents_pt_eta_phi)[...,0]*(-1), axis=1)
        # j2_reco_constituents_pt_eta_phi_sorted = np.take_along_axis(np.asarray(j2_reco_constituents_pt_eta_phi), x_j2[...,None], axis=1)

        # j1_reco_constituents = np.array(j1_reco_constituents_pt_eta_phi_sorted)
        # j2_reco_constituents = np.array(j2_reco_constituents_pt_eta_phi_sorted)
        
        # Shape of returned array: Batch Size x 2 x 100 x 3
        return np.stack([j1_constituents_pt_eta_phi, j2_constituents_pt_eta_phi], axis=1),np.stack([j1_reco_constituents_pt_eta_phi, j2_reco_constituents_pt_eta_phi], axis=1) # Do not sort based on pt

    def read_orig_and_reco_constituents_from_file(self,path):
        with h5py.File(path,'r') as f:
            constituents_orig,constituents_reco = self.read_jet_constituents_from_file(f)
            # Shape of returned arrays: Batch Size x 2 x 100 x 3
        
            return [constituents_orig, constituents_reco]
