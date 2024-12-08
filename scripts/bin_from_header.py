import sys

def extract_binary_from_header(header_file, binary_file):
    """
    Extracts binary data from a C header file written using `xxd -i` and writes it to a binary file.

    Args:
        header_file (str): Path to the (input) header file.
        binary_file (str): Path to the (output) binary file.

    Notes:
        - The function expects the header file to be written using `xxd -i` and contain a single array of unsigned char values.
    """
    h, b = open(header_file, 'r'), open(binary_file, 'wb')

    for line in h:
        if line.startswith('unsigned') or line.startswith('};'):
            continue
    
        hex_vals = line.strip().rstrip(',').split(',')
        for val in hex_vals:
            b.write(bytes([int(val, 16)]))
    
    h.close()
    b.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python bin_from_header.py <header_file> <binary_file>')
        sys.exit(1)

    extract_binary_from_header(sys.argv[1], sys.argv[2])