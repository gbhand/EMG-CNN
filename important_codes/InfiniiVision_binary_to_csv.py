#!python3

# *********************************************************
# Script to convert an InfiniiVision oscilloscope binary
# file to CSV format waveform files.
# *********************************************************

# =========================================================
# Import Modules
# =========================================================
import sys
import re
import string
import struct

# ---------------------------------------------------------
# Variables.
# ---------------------------------------------------------
waveform_type_dict = {
    0 : "Unknown",
    1 : "Normal",
    2 : "Peak Detect",
    3 : "Average",
    4 : "Horizontal Histogram",
    5 : "Vertical Histogram",
    6 : "Logic",
}
buffer_type_dict = {
    0 : "Unknown data",
    1 : "Normal 32-bit float data",
    2 : "Maximum float data",
    3 : "Minimum float data",
    4 : "Time float data",
    5 : "Counts 32-bit float data",
    6 : "Digital unsigned 8-bit character data",
}
units_dict = {
    0 : "Unknown",
    1 : "Volts",
    2 : "Seconds",
    3 : "Constant",
    4 : "Amps",
    5 : "dB",
    6 : "Hz",
}
hex_to_binary_dict = {
    "0" : "0000",
    "1" : "0001",
    "2" : "0010",
    "3" : "0011",
    "4" : "0100",
    "5" : "0101",
    "6" : "0110",
    "7" : "0111",
    "8" : "1000",
    "9" : "1001",
    "a" : "1010",
    "b" : "1011",
    "c" : "1100",
    "d" : "1101",
    "e" : "1110",
    "f" : "1111",
}


# =========================================================
# Function to print and save output information.
# =========================================================
def prtsv(string):
    global msg
    msg.write("%s\n" % string)
    print(string)


# =========================================================
# Function to read 8-bit digital data from the binary data
# file.
# =========================================================
def read_8bit_digital_data(buffer_size, x_origin, x_increment, label, segment_index):
    if segment_index == 0:
        csv_output_file = re.sub("\.bin", "_%s.csv" % label, sys.argv[1])
    else:
        csv_output_file = re.sub("\.bin", "_segment-%d_%s.csv" % (segment_index, label), sys.argv[1])
    csv = open(csv_output_file, "w")

    # prtsv("---------- Digital Data ----------")

    for i in range(buffer_size):
        (digital_data,) = struct.unpack('B', bin_input.read(1))
        # prtsv("Point %d, = '0x%0.2x', %E s" % (i, digital_data, x_origin + (i * x_increment)))

        hex_string = hex(digital_data)   # Returns "0xn" or "0xnn".
        if len(hex_string) == 4:   # "0xnn".
              un = hex_to_binary_dict[hex_string[2]]
              ln = hex_to_binary_dict[hex_string[3]]
        else:   # "0xn".
              un = "0000"
              ln = hex_to_binary_dict[hex_string[2]]
        csv.write("%s, %s, %s, %s, %s, %s, %s, %s, %s\n" % (x_origin + (i * x_increment), un[0], un[1], un[2], un[3], ln[0], ln[1], ln[2], ln[3], ))

    csv.close()
    prtsv("CSV waveform data saved to: %s" % csv_output_file)


# =========================================================
# Function to read 32-bit float data from the binary data
# file.
# =========================================================
def read_32bit_float_data(buffer_size, bytes_per_point, x_origin, x_increment, label, segment_index):
    if segment_index == 0:
        csv_output_file = re.sub("\.bin", "_%s.csv" % label, sys.argv[1])
    else:
        csv_output_file = re.sub("\.bin", "_segment-%d_%s.csv" % (segment_index, label), sys.argv[1])
    csv = open(csv_output_file, "w")

    # prtsv("---------- Voltage Data ----------")

    for i in range(int(buffer_size / bytes_per_point)):
        (voltage,) = struct.unpack('f', bin_input.read(bytes_per_point))
        # prtsv("Point %d, = '%+.2f' V, %E s" % (i, voltage, x_origin + (i * x_increment)))

        csv.write("%E, %f\n" % (x_origin + (i * x_increment), voltage))

    csv.close()
    prtsv("CSV waveform data saved to: %s" % csv_output_file)


# =========================================================
# Function to print data from individual waveforms in binary
# data file.
# =========================================================
def read_waveform_data(x_origin, x_increment, label, segment_index):

    prtsv("---------- Waveform Data Header ----------")

    (waveform_data_header_size,) = struct.unpack('i', bin_input.read(4))
    prtsv("Waveform Data Header Size = '%d'" % waveform_data_header_size)

    (buffer_type,) = struct.unpack('h', bin_input.read(2))
    if buffer_type in buffer_type_dict:
        prtsv("Buffer Type = '%s'" % buffer_type_dict[buffer_type])
    else:
        prtsv("Buffer Type (unknown) = '%d'" % buffer_type)
        bin_input.close()
        msg.close()
        sys.exit()

    (bytes_per_point,) = struct.unpack('h', bin_input.read(2))
    prtsv("Bytes Per Point = '%s'" % bytes_per_point)

    (buffer_size,) = struct.unpack('i', bin_input.read(4))
    prtsv("Buffer Size = '%d'" % buffer_size)

    if buffer_type == 1:   # Normal 32-bit float data.
        read_32bit_float_data(buffer_size, bytes_per_point, x_origin, x_increment, label, segment_index)
    elif buffer_type == 2:   # Maximum float data.
        label = label + "_PkMax"
        read_32bit_float_data(buffer_size, bytes_per_point, x_origin, x_increment, label, segment_index)
    elif buffer_type == 3:   # Minimum float data.
        label = label + "_PkMin"
        read_32bit_float_data(buffer_size, bytes_per_point, x_origin, x_increment, label, segment_index)
    elif buffer_type == 6:   # Digital unsigned 8-bit char data.
        read_8bit_digital_data(buffer_size, x_origin, x_increment, label, segment_index)
    else:
        buffer_bytes = bin_input.read(buffer_size)


# =========================================================
# Function to print data from individual waveforms in binary
# data file.
# =========================================================
def read_waveform():

    prtsv("---------- Waveform Header ----------")

    (waveform_header_size,) = struct.unpack('i', bin_input.read(4))
    prtsv("Waveform Header Size = '%d'" % waveform_header_size)

    (waveform_type,) = struct.unpack('i', bin_input.read(4))
    if waveform_type in waveform_type_dict:
        prtsv("Waveform Type = '%s'" % waveform_type_dict[waveform_type])
    else:
        prtsv("Waveform Type (unknown) = '%d'" % waveform_type)
        bin_input.close()
        msg.close()
        sys.exit()

    (waveform_buffers,) = struct.unpack('i', bin_input.read(4))
    prtsv("Number of Waveform buffers = '%d'" % waveform_buffers)

    (points,) = struct.unpack('i', bin_input.read(4))
    prtsv("Points = '%d'" % points)

    (count,) = struct.unpack('i', bin_input.read(4))
    prtsv("Count = '%d'" % count)

    (x_display_range,) = struct.unpack('f', bin_input.read(4))
    prtsv("X Display Range = '%E'" % x_display_range)

    (x_display_origin,) = struct.unpack('d', bin_input.read(8))
    prtsv("X Display Origin = '%E'" % x_display_origin)

    (x_increment,) = struct.unpack('d', bin_input.read(8))
    prtsv("X Increment = '%E'" % x_increment)

    (x_origin,) = struct.unpack('d', bin_input.read(8))
    prtsv("X Origin = '%E'" % x_origin)

    (x_units,) = struct.unpack('i', bin_input.read(4))
    if x_units in units_dict:
        prtsv("X Units = '%s'" % units_dict[x_units])
    else:
        prtsv("X Units = '%d'" % x_units)

    (y_units,) = struct.unpack('i', bin_input.read(4))
    if x_units in units_dict:
        prtsv("Y Units = '%s'" % units_dict[y_units])
    else:
        prtsv("Y Units = '%d'" % y_units)

    (date,) = struct.unpack('16s', bin_input.read(16))
    prtsv("Date = '%s'" % date.decode("utf-8"))

    (time,) = struct.unpack('16s', bin_input.read(16))
    prtsv("Time = '%s'" % time.decode("utf-8"))

    (frame,) = struct.unpack('24s', bin_input.read(24))
    prtsv("Frame = '%s'" % frame.decode("utf-8"))

    (waveform_label,) = struct.unpack('16s', bin_input.read(16))
    label = waveform_label.decode("utf-8").rstrip(chr(0))
    prtsv("Waveform Label = '%s'" % label)

    (time_tags,) = struct.unpack('d', bin_input.read(8))
    prtsv("Time Tags = '%E'" % time_tags)

    (segment_index,) = struct.unpack('I', bin_input.read(4))
    prtsv("Segment Index = '%d'" % segment_index)

    for i in range(waveform_buffers):
        read_waveform_data(x_origin, x_increment, label, segment_index)


# =========================================================
# Main Program
# =========================================================

if len(sys.argv) != 2:
    sys.stderr.write("Usage: python %s <binary_data_file>\n" % sys.argv[0])
    sys.exit()

# ---------------------------------------------------------
# Open message output file.
# ---------------------------------------------------------
msg_output_file = re.sub("\.bin", "_info.txt", sys.argv[1])
msg = open(msg_output_file, "w")

# ---------------------------------------------------------
# Open binary file.
# ---------------------------------------------------------
bin_input = open(sys.argv[1], "rb")

prtsv("---------- File Header ----------")

(cookie,) = struct.unpack('2s', bin_input.read(2))
prtsv("Cookie = '%s'" % cookie.decode("utf-8"))

(file_version,) = struct.unpack('2s', bin_input.read(2))
prtsv("File version = '%s'" % file_version.decode("utf-8"))

(file_size,) = struct.unpack('i', bin_input.read(4))
prtsv("File size = '%d'" % file_size)

(waveforms,) = struct.unpack('i', bin_input.read(4))
prtsv("Number of Waveforms = '%d'" % waveforms)

for i in range(waveforms):
    read_waveform()

# ---------------------------------------------------------
# Close binary file.
# ---------------------------------------------------------
bin_input.close()

# ---------------------------------------------------------
# Close message output file.
# ---------------------------------------------------------
msg.close()

# ---------------------------------------------------------
# Exit program.
# ---------------------------------------------------------
sys.exit()

