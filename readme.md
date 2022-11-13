# Traffic Violations (Make Classification)

To run the project you need to call run.sh file which is contained in the same directory. It first loads the data from s3, installs the required packages, trains model then starts the server. Also I have added .ipynb jupyter notebook which includes EDA + Analysis + Feature Importance part. <br>

Check out [notebook](https://github.com/Rajasvi/traffic-violations-make-classification/blob/main/traffic_violations.ipynb]) for analysis details. For graphs please check [pdf](https://github.com/Rajasvi/traffic-violations-make-classification/blob/main/traffic_violations.pdf]) version of report.

## Environment Settings
Please install required dependencies in order to run model.pkl using:
```
pip install -r requirements.txt
```
Otherwise bare minimum add these packages required for runnning code

- scikit-learn
- plotly-express
- pandas
- numpy
- flask

In order to train model and start server call:
```
run.sh
```
If you directly wish to start the server please use this command (just check you have all the files): <br>
```
python server.py
```
In order to make request please use request.py file which contains test json which can be modified. After that call: 
<br>

```
python request.py
```
