from jetmontecarlo.utils.partonshower_utils import *

# Defining the folder in which we store the parton shower data
class parton_shower():
    # DEBUG: could make abstract class
    # ------------------------------------
    # Event generation
    # ------------------------------------
    # DEBUG: gen_events, pretty bad method rn tbh
    def gen_events(self, num_events):
        self.num_events = num_events
        self.jet_list = gen_jets(num_events, beta=self.shower_beta,
                                 radius=self.radius, jet_type=self.jet_type,
                                 acc='LL' if self.fixed_coupling else 'MLL',
                                 cutoff=self.shower_cutoff)


    # ------------------------------------
    # Helper functions
    # ------------------------------------
    # DEBUG: writing and saving -- could be removed
    # def save_events(self, file_path=None, info=''):
    #     if file_path is not None:
    #         with open(file_path, 'wb') as f:
    #             pickle.dump(self.jet_list, f)
    #         return

    #     if self.verbose > 0:
    #         print("Saving {:.0e} parton shower events to {}...".format(
    #                                         self.num_events, str(file_path)), flush=True)
    #     with open(file_path, 'wb') as f:
    #         pickle.dump(self.jet_list, f)
    #     if self.verbose > 0:
    #         print("Parton shower events saved!", flush=True)

    # def load_events(self, num_events, info=''):
    #     self.num_events = num_events
    #     file_path = self.showerfile_path(info=info)
    #     if self.verbose > 0:
    #         print("Loading {:.0e} parton shower events from {}...".format(
    #                                         self.num_events, str(file_path)),
    #               flush=True)
    #     with open(file_path, 'rb') as f:
    #         self.jet_list = pickle.load(f)
    #     if self.verbose > 0:
    #         print("Parton shower events loaded!", flush=True)


    # ------------------------------------
    # Init
    # ------------------------------------
    def __init__(self, fixed_coupling, shower_cutoff=None, shower_beta=1,
                 radius=1., jet_type='quark',
                 num_events=0, verbose=1):
        # Initializing shower parameters
        self.fixed_coupling = fixed_coupling
        self.shower_cutoff = shower_cutoff if shower_cutoff is not None\
                             else 1e-10 if fixed_coupling\
                             else MU_NP
        self.shower_beta = shower_beta
        self.verbose = verbose

        self.radius = radius
        self.jet_type = jet_type

        self.jet_list = None

        # Generating events
        if num_events > 0:
            self.gen_events(num_events)
