# computational-intelligence
Automated robust first peak detection in a time signal using computational intelligence
**Peak Detection**
Goal is to detect peak in given signal(s) in directory \dataset with computation intelligence and not just signal processing.
**Approach A**
read from xlsx into dataframe

calculate fft and take complex conjugate to eleminate imaginary value

discarding any frequency below power 1.5 and calculate inverse fft

using hilbert transformation to calculate upper envelope

plot orignal noisy signal (code commented)

plot FFT of noisy and filtered signal (code commented)

plot filtered signal with upper envelope and peaks marked as 'x' (code commented)

save the real value of peaks

group the peaks in chunk of n_slice = 62

create a label as ((len(df.index),len(df.columns)//n_slice)) and mark the cell as 1 where the peak is present

create a 1D CNN network and put in the input with label

save the modeled CNN netowork for prediction

give test input and get peak
