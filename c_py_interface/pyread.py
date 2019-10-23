import ctypes
import numpy
from numpy.ctypeslib import ndpointer

#### Path to the library to be used to read the data ####
libpath = '/home/pthom001/ML_JLAB/libread.so'

## Opens the library
reader = ctypes.CDLL(libpath)

####################################################
## Python wrapper for the get_array_size function ##
####################################################

#### The arguments passed to the c function are two int* ####
reader.get_array_size.argtypes = ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)

#### The function is void ####
reader.get_array_size.restype = None

#### The two variables passed to the C function ####
rows = ctypes.c_int()
cols = ctypes.c_int()

#### Pass those variables by reference to retrieve their values on return and call it####
reader.get_array_size(ctypes.byref(rows),ctypes.byref(cols))


####################################################
## Python wrapper for the read_data function      ##
####################################################

#### The function takes no arguments ####
reader.read_data.argtype = None


#### This C function returns an array of doubles the size of which is already known by calling the get_array_size function ####
#### Data  returned in a numpy array                                                                                       ####
reader.read_data.restype =  ndpointer(dtype=ctypes.c_double,flags='C_CONTIGUOUS',shape=(rows.value,cols.value,))

####################################################
## Python wrapper for the release_data function   ##
####################################################

#### Function parameter is a pointer to the data to release ####
reader.release.argtype = ctypes.POINTER(ctypes.c_double)

#### Void function ####
reader.release.restype = None

#### Call the read data function ####
val = reader.read_data()

#### Print data to say that they are correct ####
print(val)

#### Free memory by calling the release_data function ####
reader.release(numpy.ctypeslib.as_ctypes(val))