import oandapy
import csv
import json
import pandas as pd

instrument = "USD_JPY"
granularity = "H1"

o = oandapy.API(environment='practice', 
		access_token='93e70bc4fa445648cbb2795ebf6d1a70-c5bc1eb094b3d141b81ac5987d970dae')

response = [o.get_history(instrument=instrument, granularity=granularity, 
							count=5000, candleFormat='midpoint')]

for i in range(16):
	print ("Request {} received.".format(i))
	start_time = response[i]['candles'][0]['time']
	start_time = start_time[:start_time.find('.')]
	response.append(o.get_history(instrument=instrument, granularity=granularity, 
					count=5000, candleFormat='midpoint', end=start_time))

closeMid, highMid, lowMid, openMid, volume, time = [], [], [], [], [], []
for resp in reversed(response):
	for i in resp['candles']:
		if i['complete']:
			t = i['time']
			if int(t[:4]) < 2005:
				continue
			time.append(t[:t.find('.')])
			closeMid.append(i['closeMid'])
			highMid.append(i['highMid'])
			lowMid.append(i['lowMid'])
			openMid.append(i['openMid'])
			volume.append(i['volume'])
			

df = pd.DataFrame(data={"closeMid": closeMid, "highMid": highMid, 
					"lowMid": lowMid, "openMid": openMid, 
					"volume": volume}, index=time)

df.to_csv("data/raw_candles.csv", sep=',')




