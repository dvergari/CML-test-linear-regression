import netron, os

port = int(os.environ['CDSW_APP_PORT'])

netron.start('mobilenet.h5', port)