# TNW-CATE method

This repository contains the code base for "Heterogeneous Treatment Effect with Trained Kernels of the Nadaraya-Watson regression" article. The program is written in python 3 using the Tensorflow framework.

To satisfy the dependencies use pip and requirements.txt:

```python
pip install -r requirements.txt
```

The scripts are connected with numerical experiments. To get data for the alpha dependency, run alpha_test.py. Control size dependency - size_test.py, treatment part dependency - part_test.py. The dictionaries with results will be serialized to res_dicts folder, with drawing_script.py it is possible to make figures out of ones. 

To generate results for the table use table_script.py (the easiest way to check if everything is correct). 

All experiment's params are located in the top of the scripts, to change function for the result generation edit the row with "cur_setup" variable.

