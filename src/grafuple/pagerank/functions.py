""" Module containing the functions f:Vert(G)-->R for use in graphdualvs.  Creates the dictionary
    Functions with (key,value) pairs given by (node type, set of functions to be applied to nodes of that type)

"""

import numpy as np

# list node types
nodeTypes = ['CLRLiteral','CLRVariableWithInitializer','ThrowOp','ArrayIndex','Return','ArgumentReference',
             'StoreLocal','UnaryOp','FnPtrObj','Assignment','Dereference','continue','CLRArray','ClassRef',
             'break','AddressOf','TypeCast','Call','BinaryOp','StoreArg','LocalVar','NullRef','PInvokeCall',
             'TypeTest','Entrypoint','CtorCall','FieldReference']

# instantiate dictionary to hold node types and functions for each node type
Functions = {key:[] for key in nodeTypes}

# indicator function for each type
def ExpectedType(*args):
    return 1.0
map(lambda x: Functions[x].append(ExpectedType),nodeTypes)

          
# if node is of a certain type return transformed string based on that node type
def CLRVariable(*args):
    return np.log10(max(1,abs(hash(args[0]['varType']))))
Functions['CLRVariableWithInitializer'].append(CLRVariable)

def BinaryOp(*args):
    return np.log10(max(1,abs(hash(args[0]['op']))))
Functions['BinaryOp'].append(BinaryOp)

def CtorCallctorType(*args):
    return np.log10(max(1,abs(hash(args[0]['ctorType']))))
Functions['CtorCall'].append(CtorCallctorType)

def FieldReference(*args):
    return np.log10(max(1,abs(hash(args[0]['fieldName']))))
Functions['FieldReference'].append(FieldReference)

def CLRLiteral(*args):
    return np.log10(max(1,abs(hash(args[0]['value']))))
Functions['CLRLiteral'].append(CLRLiteral)

def CallfnName(*args):
    return np.log10(max(1,abs(hash(args[0]['fnName']))))
Functions['Call'].append(CallfnName)

def CLRArrayelemType(*args):
    return np.log10(max(1,abs(hash(args[0]['elemType']))))
Functions['CLRArray'].append(CLRArrayelemType)

def FnPtrObjname(*args):
    return np.log10(max(1,abs(hash(args[0]['name']))))
Functions['FnPtrObj'].append(FnPtrObjname)

def TypeTesttestedType(*args):
    return np.log10(max(1,abs(hash(args[0]['testedType']))))
Functions['TypeTest'].append(TypeTesttestedType)

def ClassRefname(*args):
    return np.log10(max(1,abs(hash(args[0]['name']))))
Functions['ClassRef'].append(ClassRefname)

def TypeCast(*args):
    return np.log10(max(1,abs(hash(args[0]['castedType']))))
Functions['TypeCast'].append(TypeCast)

def CLRArraysize(*args):
    return np.log10(max(1,abs(hash(args[0]['elemType']))))
Functions['CLRArray'].append(CLRArraysize)



# if node type is of a certain type return numerical value based on that node type
def NumPass2Call(*args):           
    return len(args[0]['arguments'])
Functions['Call'].append(NumPass2Call)  

def AddressOf(*args):
    return float(args[0]['expr'])
Functions['AddressOf'].append(AddressOf)

def ThrowOpexpr(*args):
    return float(args[0]['expr'])
Functions['ThrowOp'].append(ThrowOpexpr)

def UnaryOpexpr(*args):
    return float(args[0]['expr'])
Functions['UnaryOp'].append(UnaryOpexpr)

def StoreLocallocalIdx(*args):
    return int(args[0]['localIdx'])
Functions['StoreLocal'].append(StoreLocallocalIdx)

def StoreLocalvalue(*args):
    return float(args[0]['value'])
Functions['StoreLocal'].append(StoreLocalvalue)

def Returnvalue(*args):
    V = args[0]['value']
    if type(V) is dict:
        return 0
    else:
        return float(V)
Functions['Return'].append(Returnvalue)








