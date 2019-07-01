import os

if os.path.exists('/media/ephemeral0/RandomForest.pkl'):
    DEFAULT_DOTNET_MODEL_PATH = '/media/ephemeral0/RandomForest.pkl'
else:
    DEFAULT_DOTNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'RandomForest.pkl')

if os.path.exists('/opt/infinitity/dotnet/CLRParserApp/CLRParserApp.exe'):
    DEFAULT_DECOMPILER_PATH = '/opt/infinitity/dotnet/CLRParserApp/CLRParserApp.exe'
else:
    DEFAULT_DECOMPILER_PATH = os.path.join(os.path.dirname(__file__), 'lib', 'CLRParserApp.exe')

DOTNET_MODEL_PATH = os.environ.get('DOTNET_MODEL_PATH', DEFAULT_DOTNET_MODEL_PATH)
DECOMPILER_PATH = os.environ.get('DECOMPILER_PATH', DEFAULT_DECOMPILER_PATH)