## Traffic Violations (Make Classification)

To run the project you need to call run.sh file which is contained in the same directory. It first loads the data from s3, installs the required packages, trains model then starts the server. Also I have added .ipynb jupyter notebook which includes EDA + Analysis + Feature Importance part. <br>

If you directly wish to start the server please use this command (just check you have all the files): <br>
```
python server.py
```
<br>
In order to make request please use request.py file which contains test json which can be modified. After that call: <br>
```
python request.py
```
