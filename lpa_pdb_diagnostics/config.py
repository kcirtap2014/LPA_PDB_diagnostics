import os

#Default value of the result_path configuration setting
result_path = os.getcwd() + "/result/"

if not os.path.exists( result_path ):
    print ("Result path %s is created" %result_path)
    os.makedirs( result_path )

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
