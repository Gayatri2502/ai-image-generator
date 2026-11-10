"""Create a tiny original geometric dataset for testing the training command."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw

def create(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    records = []
    for i, color in enumerate(['#e47062','#73adbc','#8da970','#be96d0']):
        image = Image.new('RGB', (128, 128), '#f5f0e8')
        ImageDraw.Draw(image).ellipse((20,20,108,108), fill=color)
        name = f'circle-{i}.png'; image.save(directory / name)
        records.append({'file_name': name, 'text': f'a minimal flat illustration of a colored circle on a cream background, color {color}'})
    (directory / 'metadata.jsonl').write_text('\n'.join(json.dumps(r) for r in records) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('directory'); create(parser.parse_args().directory)
