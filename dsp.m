highdir = "high\";
lowdir = "low\";

mkdir('high');
mkdir('low');

delete('high\*.csv');
delete('low\*.csv');

raw_high = high;
raw_low = none;

Fs = 4000;
len = 40; % window length
minfreq = 5;
maxfreq = 30;
% filter design

[b1,a1] = butter(6,15/(Fs/2),'high'); % highpass filter cutoff at 15Hz

wo = 60/(Fs/2);  bw = wo/35; % 60 Hz filter
[b2,a2] = iirnotch(wo,bw); % 60 Hz filter

w1 = 120/(Fs/2);  bw1 = w1/35; % 120 Hz filter
[b3,a3] = iirnotch(w1,bw1); %  120 Hz filter

[b4,a4] = butter(3,140/(Fs/2),'low');

w2 = 180/(Fs/2);  bw2 = w2/35;
[b5,a5] = iirnotch(w2,bw2);

len_high = length(raw_high) - mod(raw_high, len);
raw_high = raw_high(1:len_high);

raw = raw_high;
EMG1 = filter(b1, a1, raw);
EMG2 = filter(b2, a2, EMG1);
EMG3 = filter(b3, a3, EMG2);
EMG4 = filter(b4, a4, EMG3);
EMG5 = filter(b5, a5, EMG4);
EMG53 = filter(b3, a3, EMG5);
EMG52 = filter(b2, a2, EMG53);

% filt_high = EMG52;
filt_high = raw_high;   %TEMP REMOVE wait wow this actually works

index = 1;
for win = 1:(len_high / len)
    curr_data = filt_high(index:index+len);
    pxx = st(curr_data,minfreq,maxfreq,Fs);
    writematrix(abs(pxx), "high\" + win + ".csv"); 
    index = index + len;
end


len_low = length(raw_low) - mod(raw_low, len);
raw_low = raw_low(1:len_low);

raw = raw_low;
EMG1 = filter(b1, a1, raw);
EMG2 = filter(b2, a2, EMG1);
EMG3 = filter(b3, a3, EMG2);
EMG4 = filter(b4, a4, EMG3);
EMG5 = filter(b5, a5, EMG4);
EMG53 = filter(b3, a3, EMG5);
EMG52 = filter(b2, a2, EMG53);

% filt_low = EMG52;
filt_low = raw_low;     %TEMP REMOVE

index = 1;
for win = 1:(len_low / len)
    curr_data = filt_low(index:index+len);
    pxx = st(curr_data,minfreq,maxfreq,Fs);
    writematrix(abs(pxx), "low\" + win + ".csv"); 
    index = index + len;
end

% disp(win);
% disp(index);
