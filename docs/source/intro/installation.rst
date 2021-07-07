Installation
============

Install InstanceLib
----------------

InstanceLib requires having Python 3.8 or higher installed. 

Install the InstanceLib with `pip` by running the following command in your terminal:

.. code:: bash

    pip install instancelib

You then import InstanceLib in your Python code as follows: 

.. code:: python

    import instancelib as il

You are now ready to start your InstanceLib in your application!

See `Troubleshooting`_ for common problems.



Upgrade InstanceLib
----------------

Upgrade InstanceLib as follows:

.. code:: bash

    pip install --upgrade instancelib



Uninstall InstanceLib
------------------

Remove InstanceLib with

.. code:: bash

    pip uninstall instancelib

Enter ``y`` to confirm.


Troubleshooting
---------------

InstanceLib is advanced machine learning software. In some situations, you
might run into unexpected behavior. See below for solutions to
problems.

Unknown Command "pip"
~~~~~~~~~~~~~~~~~~~~~

The command line returns one of the following messages:

.. code:: bash

  -bash: pip: No such file or directory

.. code:: bash

  'pip' is not recognized as an internal or external command, operable program or batch file.


First, check if Python is installed with the following command:

.. code:: bash

    python --version

If this does not return 3.8 or higher, then Python is not (correctly)
installed.

However, there is a simple way to deal with correct environment variables
by adding `python -m` in front of the command. For example:

.. code:: bash

  python -m pip install instancelib

