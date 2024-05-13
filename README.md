![i2cat](https://ametic.es/sites/default/files//i2cat_w.png)
# PHaul

Related Jira project: [MISSING]

MWI, 2024

CODEOWNERS: daniel.camps@i2cat.net, jorge.pueyo@i2cat.net

# Project status
[MISSING]
# Short description
Python implementation of PHAUL, a Deep Reinforcement Learning Agent that produces the optimal flow allocations for Integrated Access Backhaul networks. 
# Pre-requisites
- Ubuntu >= 20.04
- Python >= 3.7
- TensorFlow >= 2.6
- OpenAI Gym
- Stable-Baselines3
- conda

# How to install, build & run
After cloning the repository to local, install it using conda: \
``conda env create -f environment.yml``

To change the parameters of the agent modify the `src/cfg.yaml` file, where you can find all the available parameters. 

To run the agent execute ``python src/main.py``. The `src/main.py` file is configured to run multiple executions of the agent in parallel, using as many CPUs as required, and varying the values of the parameters used. The results of the executions are stored as `.txt` files. The script `src/utils/results_to_csv.py` can be used to generate a single CSV file contatining all the results of a single execution. 


# Source
This code has been developed within the research / innovation project [MISSING].
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No [MISSING]

This project started on 1 June 2017 and ended on 31 March 2020, funded under H2020-
EU.2.1.1 of the Horizon 2020 Framework Programme, with an overall budget of
€7.710.063,63 and an EU contribution of €6.072.367,40, coordinated by I2CAT.
More information about the grant at https://cordis.europa.eu/project/id/761508
# Copyright
This code has been developed by Fundació Privada Internet i Innovació Digital a Catalunya (i2CAT).
i2CAT is a *non-profit research and innovation centre* that  promotes mission-driven knowledge to solve business challenges, co-create solutions with a transformative impact, empower citizens through open and participative digital social innovation with territorial capillarity, and promote pioneering and strategic initiatives.
i2CAT *aims to transfer* research project results to private companies in order to create social and economic impact via the out-licensing of intellectual property and the creation of spin-offs.
Find more information of i2CAT projects and IP rights at https://i2cat.net/tech-transfer/
# License
This code is licensed under the terms AGPLv3. Information about the license can be located at [link](https://www.gnu.org/licenses/agpl-3.0.html).
If you find that this license doesn't fit with your requirements regarding the use, distribution or redistribution of our code for your specific work, please, don’t hesitate to contact the intellectual property managers in i2CAT at the following address: techtransfer@i2cat.net

Also, in the following page you’ll find more information about the current commercialization
status or other licensees: Enllaç a la pàgina web de l’asset (si existeix).
