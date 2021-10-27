Installation
============

Install InstanceLib
-------------------

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
-------------------

Upgrade InstanceLib as follows:

.. code:: bash

    pip install --upgrade instancelib



Uninstall InstanceLib
---------------------

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


First, check if Python 3.8 is installed by issuing one of the following commands:

.. code:: bash

    python --version

.. code:: bash

    python3 --version

.. code:: bash

    python3.8 --version

If this does not return 3.8 or higher, then Python is not (correctly) installed.
We recommend making a virtual environment in which you install your packages.
You can create a virtual environment `.venv` in your working directory as follows (adapt your python command if necessary).

.. code:: bash

    python -m venv .venv


