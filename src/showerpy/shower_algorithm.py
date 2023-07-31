import random

# =====================================
# Shower algorithm class
# =====================================
class ShowerAlgorithm:
    """

    Attributes
    ----------
        cutoff_scale : float
            The scale at which the shower is cut off.
        split_soft : bool
            Whether or not to split soft emissions.
        use_veto : bool
            Whether or not to use the veto algorithm for emission
            generation after generating random momentum fractions and
            angles for an emission.
            If False, emission generation uses the user defined method
            `random_emission_scale` without additional processing.
    """
    def __init__(self, cutoff_scale,
                 split_soft=False, use_veto=True):
        self.cutoff_scale = cutoff_scale
        self.split_soft = split_soft
        self.use_veto = use_veto


    def __call__(self, seed_parton, initial_scale,
                 **kwargs):  # -> List[Parton]
        """A method which takes in a seed parton and returns a jet.
        """
        if initial_scale > self.cutoff_scale:
            # Finding energy fraction and angle of the emission
            momentum_fraction, theta = self.emission_z_theta(initial_scale,
                                                             **seed_parton.metadata)
            if momentum_fraction is None:
                assert theta is None, "momentum_fraction and theta must be None together"
                return

            # Splitting the parton
            seed_parton.split(momentum_fraction, theta)
            daughter1, daughter2 = seed_parton.daughters

            # Recursively calling the shower algorithm on the daughters
            final_scale = self.get_scale(momentum_fraction, theta)
            self.__call__(daughter1, final_scale)
            # Adding an option to split or not split soft emissions
            if self.split_soft:
                self.__call__(daughter2, final_scale)

        return seed_parton.daughter_tree()


    def generation_pdf(self, momentum_fraction, theta, **kwargs):
        """The probability density function for the generation
        of the scales associated with emissions."""
        raise NotImplementedError

    def true_pdf(self, momentum_fraction, theta, **kwargs):
        """The true probability density function associated with
        the splitting of a parton."""
        raise NotImplementedError

    def get_scale(self, momentum_fraction, theta):
        """The scale associated with the momentum fraction and angle
        of an emission."""
        raise NotImplementedError


    def random_emission_scale(self, initial_scale: float,
                              **kwargs) -> float:
        """A method which returns a random emission scale
        given the initial scale."""
        raise NotImplementedError

    def random_z_and_theta(self, scale: float,
                           **kwargs) -> (float, float):
        """A method which returns a random z and theta
        given a scale."""
        raise NotImplementedError


    def emission_z_theta(self, mother_scale, **kwargs):
        """A method which returns a z and theta of an emission
        given an initial scale.

        Uses the veto algorithm if specified during class
        initialization.
        """
        accept_emission = False
        while not accept_emission:
            emission_scale = self.random_emission_scale(mother_scale,
                                                        **kwargs)
            momentum_fraction, theta = self.random_z_and_theta(emission_scale,
                                                               **kwargs)

            if not self.use_veto:
                accept_emission = True
            else:
                cut = (self.generation_pdf(momentum_fraction, theta, **kwargs)
                       / self.true_pdf(momentum_fraction, theta, **kwargs))
                if cut > 1:
                    raise ValueError("The pdf must be everywhere less than the"
                                     + "proposed pdf!")
                if cut > .6:
                    print(f"Dangerous value of {cut} for pdf ratio "
                          +"in the veto algorithm.", flush=True)

                if random.random() < cut:
                    # If we accept the emission, stop the algorithm here
                    accept_emission = True
                else:
                    # Otherwise, continue but reset the scale of the emission
                    mother_scale = emission_scale

                # The above lines of code are the soul of the veto algorithm:
                # rather than generating from scratch, as you would
                # for the von Neumann acceptance-rejection algorithm,
                # you use this scale as the scale for the next emission.
                # This correctly takes into account the exponentiation of
                # multiple emissions, as described in the Pythia manual:
                # https://arxiv.org/pdf/hep-ph/0603175.pdf#page=66&zoom=150,0,240

        return momentum_fraction, theta
