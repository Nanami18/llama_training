import json
import zstandard as zstd
import io
import argparse

def decompress_zst(input_file, output_file):
    
    if output_file is None:
        output_file = input_file[:-4]

    dctx = zstd.ZstdDecompressor()

    with zstd.open(input_file, 'rb', dctx=dctx) as compressed_file:
        with open(output_file, 'wt', encoding='utf-8') as decompressed_file:
            # Create a decompression stream
            with io.TextIOWrapper(compressed_file) as reader:
                for line in reader:
                    decompressed_file.write(line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str)
    
    args = parser.parse_args()

    decompress_zst(args.input_file, args.output_file)