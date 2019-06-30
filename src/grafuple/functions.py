import numpy as np
import hashlib
from functools import partial



my_hash = lambda s: int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8
scaled_hash = lambda s: np.log10(max(1,abs(my_hash(s))))

def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
        

def hash_type(key,node_type,node):
    value = scaled_hash(node[key]) if ('type' in list(node.keys())) and (node['type']==node_type) and (key in list(node.keys())) else 0
    return value


functions = {}
functions['CLRVariableWithInitializer'] = partial(hash_type,'varType','CLRVariableWithInitializer')
functions['BinaryOp'] = partial(hash_type,'op','BinaryOp')
functions['CtorCall'] = partial(hash_type,'ctorType','CtorCall')
functions['FieldReference'] = partial(hash_type,'fieldName','FieldReference')
functions['CLRLiteral'] = partial(hash_type,'value','CLRLiteral')
functions['Call'] = partial(hash_type,'fnName','Call')
functions['CLRArray'] = partial(hash_type,'elemType','CLRArray')
functions['FnPtrObj'] = partial(hash_type,'name','FnPtrObj')
functions['TypeTest'] = partial(hash_type,'testedType','TypeTest')
functions['ClassRef'] = partial(hash_type,'name','ClassRef')
functions['TypeCast'] = partial(hash_type,'castedType','TypeCast')


def NumPass2Call(node):
    value = len(node['arguments']) if ('type' in list(node.keys())) and (node['type']=='Call') and ('arguments' in list(node.keys())) else 0
    return value
functions['NumPass2Call'] = NumPass2Call

def AddressOf(node):
    value = float(node['expr']) if ('type' in list(node.keys())) and (node['type']=='AddressOf') and ('expr' in list(node.keys())) else 0
    return value
functions['AddressOf'] = AddressOf

def ThrowOpexpr(node):
    value = float(node['expr']) if ('type' in list(node.keys())) and (node['type']=='ThrowOp') and ('expr' in list(node.keys())) else 0
    return value
functions['ThrowOpexpr'] = ThrowOpexpr

def UnaryOpexpr(node):
    value = float(node['expr']) if ('type' in list(node.keys())) and (node['type']=='UnaryOp') and ('expr' in list(node.keys())) else 0
    return value
functions['UnaryOpexpr'] = UnaryOpexpr

def StoreLocalIdx(node):
    value = float(node['localIdx']) if ('type' in list(node.keys())) and (node['type']=='StoreLocal') and ('localIdx' in list(node.keys())) else 0
    return value
functions['StoreLocalIdx'] = StoreLocalIdx

def StoreLocalValue(node):
    value = float(node['value']) if ('type' in list(node.keys())) and (node['type']=='StoreLocal') and ('value' in list(node.keys())) else 0
    return value
functions['StoreLocalValue'] = StoreLocalValue

def Return_value(node):
    value = node['value'] if ('type' in list(node.keys())) and (node['type']=='Return') and ('value' in list(node.keys())) and (type(node['value'])==int) else 0
    return value 
functions['Return_value'] = Return_value


nodeTypes = ['CLRLiteral','CLRVariableWithInitializer','ThrowOp','ArrayIndex','Return','ArgumentReference',
             'StoreLocal','UnaryOp','FnPtrObj','Assignment','Dereference','continue','CLRArray','ClassRef',
             'break','AddressOf','TypeCast','Call','BinaryOp','StoreArg','LocalVar','NullRef','PInvokeCall',
             'TypeTest','Entrypoint','CtorCall','FieldReference']

def Expected_type(s,node):
    value = int(node['type']==s) if ('type' in list(node.keys())) else 0
    return value
exp_type_funcs = {}
for t in nodeTypes:
    exp_type_funcs['%s_' % t] = partial(Expected_type,t) 


functions.update(exp_type_funcs)













