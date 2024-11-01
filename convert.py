import pandas as pd
import numpy as np

# Load the text file
input_file = 'wifi_db/noisy_dataset.txt'
data = np.loadtxt(input_file, delimiter=' ', dtype=str)
print(data)

# Convert the text file to a Pandas DataFrame
df = pd.DataFrame(data)
print(df)

# Save the DataFrame to a CSV file
output_file = 'wifi_db/noisy_dataset.csv'
df.to_csv(output_file, index=False)