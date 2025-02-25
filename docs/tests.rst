Unit Testing
=====

Unit tests are run automatically using GitHub actions, but you can run them locally by following the instructions below.

Running HDP Unit Tests
----------------------

Unit tests are provided in :code:`hdp/tests` and can be run using `PyTest <https://docs.pytest.org/en/stable/>`_. A developer environment configuration is provided in :code:`environment.yml` to quickly run these tests. Start by cloning the GitHub repository and navigating into the directory:

.. code-block:: console

   git clone https://github.com/AgentOxygen/HDP.git
   cd HDP

Then create the conda environment and load it into the shell:

.. code-block:: console

   conda env create --file=environment.yml
   conda activate hdp_dev

Install the HDP from source and then run the tests using PyTest:

.. code-block:: console

   pip install -e .
   pytest hdp/tests

