# import necessary libraries
import random
import string


class Poly:
    """ The polymorphic class to generate randomly a table of Keys (ToKs)
    This keys will be used later on for each encryption round or per each client."""

    def __init__(self, table_size, global_key_aes):
        """
        :param int tableSize: to define number of keys to be generated
        :param bytestring globalKeyAES: the global key to encrypt ToKs
        """

        self.tableSize = table_size
        self.globalKeyAES = global_key_aes

    def generate_table_of_keys(self, table_size):
        """ :return: function should return the table of keys."""
        ToKs = [[] for x in range(table_size)]
        for i in range(table_size):
            ToKs[i] = [i,
                       str.encode(''.join(random.choice(string.ascii_letters + string.digits) for i in range(32))),
                       False]
        return ToKs

    def abc(self):
        print("OK")
