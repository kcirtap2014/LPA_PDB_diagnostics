import os

class ResultPath:

    "Class that manages the result path"
    def __init__( self, path_name = None ):
        """"
        initialize result path

        """
        if path_name is None:
            self.result_path = os.getcwd() + "/result/"
        else:
            self.result_path = path_name

        if not os.path.exists(self.result_path):
            print "Result path %s is created" %self.result_path
            os.makedirs(self.result_path)
