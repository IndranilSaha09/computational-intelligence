# computational-intelligence
Automated robust first peak detection in a time signal using computational intelligence

**Peak_detection:**
Goal is to detect peak in given signal(s) in directory \dataset with computation intelligence and not just signal processing.

**Approach A:**
1.	read from xlsx into dataframe
2.	calculate fft and take complex conjugate to eleminate imaginary value
3.	discarding any frequency below power 1.5 and calculate inverse fft
4.	using hilbert transformation to calculate upper envelope
5.	plot orignal noisy signal (code commented)
6.	plot FFT of noisy and filtered signal (code commented)
7.	plot filtered signal with upper envelope and peaks marked as 'x' (code commented)
8.	save the real value of peaks
9.	group the peaks in chunk of n_slice = 62
10.	create a label as ((len(df.index),len(df.columns)//n_slice)) and mark the cell as 1 where the peak is present
11.	create a 1D CNN network and put in the input with label
12.	save the modeled CNN netowork for prediction
13.	give test input and get peak

**Approach B:**
1.	read test input into dataframe
2.	using scipy.signal.find_peak() and get peak

**Compare results from approach A and approach B and display results in web based UI **
•	Run app.py to get web based UI to upload sample file and get results

**How to run the experiment**
1.	Run pip install -r requirement.txt and get all libs

2.	Run python model_train.py so we have the trained model save

3.	Run python app.py to get web based UI and upload sample input from \static\files and get comparision
