from pylsl import StreamInlet, resolve_stream 
# 00:07:80:89:81:9B
# EC:1B:BD:62:F6:9A

import os 
import pandas as pd 

streams = resolve_stream('name','StreamName') 

inlet = StreamInlet(streams[0]) 

output_path='./data.csv' 
df = pd.DataFrame(columns=['val'])

while True:
    sample,timestamp = inlet.pull_sample() 
    df.append({'val':sample},ignore_index=True) 
    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path)) 