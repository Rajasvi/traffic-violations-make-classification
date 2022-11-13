import requests

url = "http://localhost:5000/predict"
test = {
    "Ticket number": 1105461453,
    "Issue Date": "2015-09-15T00:00:00",
    "Issue time": 115.0,
    "Meter Id": "null",
    "Marked Time": "null",
    "RP State Plate": "CA",
    "Plate Expiry Date": 200316.0,
    "VIN": "null",
    "Make": "CHEV",
    "Body Style": "PA",
    "Color": "BK",
    "Location": "GEORGIA ST\/OLYMPIC",
    "Route": "1FB70",
    "Agency": 1.0,
    "Violation code": "8069A",
    "Violation Description": "NO STOPPING\/STANDING",
    "Fine amount": 93.0,
    "Latitude": 99999.0,
    "Longitude": 99999.0,
    "popularMakeorNot": 1,
}

r = requests.post(
    url,
    json=test,
)
print(r.json())
