""" Utility fcns
"""


def flatten_dict(d):
    """ Takes python dictionary of dictionaries and flattens to single level - any element in the result can be
        written as (key,value)

        Args:
            d (dict): python dictionary

        Returns:
            d_flat (dict): python dictionary

    """

    def expand(key, value):
        """ Core helper function to append contents of (key,value) where value is itself a dictionary to the
            aggregate list of key value pairs

            Args:
                key (any hashable type): index set for values in dictionary
                value (any type): value = d[key]

            Returns:
                [(key,value)] (list): list of key value pairs

        """

        # if value is itself a dictionary, then recurse.  else return the key value pair
        if isinstance(value, dict):
            return [ (key + '.' + k, v) for k, v in flatten_dict(value).items() ]
        else:
            return [ (key, value) ]

    # aggregate all key value pairs and convert to dictionary
    items = [ item for k, v in d.items() for item in expand(k, v) ]
    d_flat = dict(items)

    return d_flat
