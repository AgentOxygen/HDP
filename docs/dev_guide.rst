Developer Guide
===============

Thank you for taking interesting in further developing the HDP! To get started, first clone the repository:

.. code-block:: console

   git clone git@github.com:AgentOxygen/HDP.git
   cd HDP

We provide a few development workflow options:

.. _docker_setup:

Docker (preferred)
------------------

Build the docker container:

.. code-block:: console

   docker build --rm -t hdp .

To run the full testing suite:

.. code-block:: console
   
   docker run -v .:/project -it hdp

To run specific tests (the workflow for example):

.. code-block:: console
   
   docker run -v .:/project -it hdp pytest hdp/tests/test_workflow.py

To generate a live view of the documentation (web server hosted at ``localhost:7000``):

.. code-block:: console
   
   docker run -v .:/project -p 7000:7000 -it hdp sphinx-autobuild docs/ docs/_build/ --host 0.0.0.0 --port 7000


Conda Environment
-----------------

.. code-block:: console

   conda env create --file=environment.yml
   conda activate hdp_dev
   pip install -e .
   pytest hdp/tests

Existing Environment
--------------------

.. code-block:: console

   pip install -r requirements.txt
   pip install -e .
   pytest hdp/tests

