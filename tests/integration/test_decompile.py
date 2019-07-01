# /usr/bin/env python
# coding=utf-8
from mock import patch
import json
from nose import tools as nt
import subprocess
import signal
import tempfile
import shutil


def test_decompile():
    """ Unit test for decompiling .NET.  Compiles a hard-coded HelloWorld C# program,
        calls the decompiler, and compares the output json to known correct output.  

        Args:
            path2decompiler (str): path to decompiler app

        Returns:
            json_data (dict): json containing decompilation data

    """

    class Alarm(Exception):
        """ Dummy class to handle timeout exception

        """
        pass


    def alarm_handler(signum, frame):
        """ Method to instantiate object to handle timeout exception

        """
        raise Alarm


    # HelloWorld C# program
    program = 'public class Hello\n{\n\tpublic static void Main()\n\t{\n\t\tSystem.Console.WriteLine("Hello, World!");\n\t}\n}'

    # create temporary directory
    tmpdir = tempfile.mkdtemp()

    # write C# source code to text file for later compilation
    with open(tmpdir+'/HelloWorld.cs','w') as f:
        f.write(program)

    # call mono compiler via subprocess module and write result to stdout
    proc = subprocess.Popen(
        [
            'mcs',
            tmpdir+'/HelloWorld.cs'

        ],

        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # write to stdout_buffer
    stdout_buffer, stderr_buffer = proc.communicate()

    # throw exception if decompilation takes longer than 1 second
    TIMEOUT=1

    # path to decompiler - to be open  sourced (does not currently exist in repo)
    path2decompiler = '../../bin/CLRParserApp.exe'

    # decompile compiled program and write to stdout
    proc = subprocess.Popen(
        [
            # use mono to call .NET app - use single flag to simplify output
            'mono',
            path2decompiler,
            "--output:json",
            "--sdfg",
            tmpdir+'/HelloWorld.exe'
        ],

        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # instantiate exception objects
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(TIMEOUT)

    # exception handling
    try:
        stdout_buffer, stderr_buffer = proc.communicate()
        signal.alarm(0)
    except Alarm:
        proc.kill()
        return None, 'timeout'
    if len(stdout_buffer.strip()) == 0:
        return None, 'empty_output'

    decompiled = json.loads(stdout_buffer)

    # remove temporary directory
    shutil.rmtree(tmpdir)

    first_test = decompiled['functions'][u'1'][u'sdfg:json']['nodes'][u'1']['type']=='Entrypoint'
    second_test = decompiled['functions'][u'2'][u'sdfg:json']['nodes'][u'4']['type']=='LocalVar'

    # unit test with known output
    nt.assert_true(first_test and second_test)
