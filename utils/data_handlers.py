import numpy as np
from sarewt.data_reader import CaseDataReader
from vande.util.data_generator import CaseDataGenerator,mask_training_cuts,constituents_to_input_samples,events_to_input_samples

import h5py
class PathManager():
    def __init__():
        pass

''' 
We can inherit from the original Case Data Reader and modify only the part where pt,eta,phi is converted to px,py,pz
For the normalizing flow, we don't need to convert 

'''
class CMSDataGenerator(CaseDataGenerator):
    def __init__(self, path, sample_part_n=1e4, sample_max_n=None, **cuts):
        super(CMSDataGenerator).__init__(self, path, sample_part_n=1e4, sample_max_n=None, **cuts)
    def __call__(self): # -> generator object yielding np.ndarray, np.ndarray
        '''
            generate single(!) data-sample (batching done in tf.Dataset)
        '''
        
        # create new file data-reader, each time data-generator is called (otherwise file-data-reader generation not reset)
        generator = CMSDataHandler(self.path).generate_event_parts_from_dir(parts_n=self.sample_part_n, **self.cuts)

        samples_read_n = 0
        # loop through whole dataset, reading sample_part_n events at a time
        for constituents, features in generator:
            samples = events_to_input_samples(constituents[:,:,:,:3], features)
            indices = list(range(len(samples)))
            samples_read_n += len(samples)
            while indices:
                index = indices.pop(0)
                next_sample = samples[index] #.copy() 
                yield next_sample
            if self.sample_max_n is not None and (samples_read_n >= self.sample_max_n):
                break
        
        print('[DataGenerator]: __call__() yielded {} samples'.format(samples_read_n))
        generator.close()

class CMSDataHandler(CaseDataReader):
    def __init__(self,path):
        super(CMSDataHandler).__init__(self,path)
        self.jet_orig_constituents_key='jetOrigConstituentsList'
        self.jet_reco_constituents_key='jetConstituentsList'    
        

    def read_events_from_file(self, fname=None, **cuts): # -> np.ndarray, np.ndarray
        fname = fname or self.path
        constituents_orig, features = self.read_constituents_and_dijet_features_from_file(fname) # -> np.ndarray, np.ndarray
        try:
            constituents, features = self.read_constituents_and_dijet_features_from_file(fname) # -> np.ndarray, np.ndarray
            if cuts:
                constituents, features = self.make_cuts(constituents, features, **cuts) # -> np.ndarray, np.ndarray
        except OSError as e:
            print("\n[ERROR] Could not read file ", fname, ': ', repr(e))
        except IndexError as e:
            print("\n[ERROR] No data in file ", fname, ':', repr(e))
        except Exception as e:
            print("\nCould not read file ", fname, ': ', repr(e))
        return np.asarray(constituents), np.asarray(features)


    def extend_by_file_content(self, constituents, features, fname, **cuts):
        cc, ff = self.read_events_from_file(fname, **cuts)
        constituents.extend(cc)
        features.extend(ff)
        return constituents, features

    def generate_event_parts_by_num(self, parts_n, flist, **cuts):
        # keeping data in lists for performance
        constituents_concat = []
        features_concat = []

        for i_file, fname in enumerate(flist):
            constituents_concat, features_concat = self.extend_by_file_content(constituents_concat, features_concat, fname, **cuts)

            while (len(features_concat) >= parts_n): # if event sample size exceeding max size or min n, yield next chunk and reset
                constituents_part, constituents_concat = constituents_concat[:parts_n], constituents_concat[parts_n:] # makes copy of *references* to ndarrays 
                features_part, features_concat = features_concat[:parts_n], features_concat[parts_n:]
                yield (np.asarray(constituents_part), np.asarray(features_part)) # makes copy of all data, s.t. yielded chunk is new(!) array (since *_part is a list) => TODO: CHECK!
        
        # if data left, yield it
        if features_concat:
            yield (np.asarray(constituents_concat), np.asarray(features_concat))


    def generate_event_parts_from_dir(self, parts_n=None, parts_sz_mb=None, **cuts):
        '''
        file parts generator
        yields events in parts_n (number of events) or parts_sz_mb (size of events) chunks
        '''
        
        # if no chunk size or chunk number given, return all events in all files of directory
        if not (parts_sz_mb or parts_n):
            return self.read_events_from_dir(**cuts)

        flist = self.get_file_list()

        if parts_n is not None:
            gen = self.generate_event_parts_by_num(int(parts_n), flist, **cuts)
        else: 
            gen = self.generate_event_parts_by_size(parts_sz_mb, flist, **cuts)

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
        
        return np.stack([j1_constituents_pt_eta_phi, j2_constituents_pt_eta_phi], axis=1),np.stack([j1_reco_constituents_pt_eta_phi, j2_reco_constituents_pt_eta_phi], axis=1) # Do not sort based on pt

    def read_orig_and_reco_constituents_from_file(self,path):
        with h5py.File(path,'r') as f:
            #features = np.array(f.get(self.jet_features_key))
            #print(self.jet_features_key)
            #print(path)
            constituents_orig,constituents_reco = self.read_jet_constituents_from_file(f)
            #print(features)
            #print("WTF")
            #import pdb;pdb.set_trace()
            return [constituents_orig, constituents_reco]
