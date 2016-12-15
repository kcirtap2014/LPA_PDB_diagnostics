import os

#Default value of the result_path configuration setting
result_path = os.getcwd() + "/result/"

if not os.path.exists( result_path ):
    print "Result path %s is created" %result_path
    os.makedirs( result_path )
