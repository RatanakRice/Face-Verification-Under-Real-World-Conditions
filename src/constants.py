"""
Shared parameters used by more than one script, so they only need changing
in one place. See the report's Parameter Settings section for how each
value was chosen.
"""

# Lighting correction: gamma values tested (gamma < 1 brightens, > 1 darkens).
GAMMA_VALUES = [0.4, 1.8, 2.5]

# Occlusion box size used for the reported results, as a fraction of image
# width/height (passed to occlusion.py as --size-min/--size-max).
OCCLUSION_SIZE = 0.40
