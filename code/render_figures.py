"""Render all twelve approved figures from packaged source tables and SVG assets."""
import argparse,os,subprocess,sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New or separate render directory')
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    output=args.output.resolve()
    if output==here.parent/'figures':parser.error('Use a separate output directory to preserve approved exports')
    output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(CGC_FIGURE_OUTPUT=str(output),PYTHONIOENCODING='utf-8')
    for script in ['figure_1.py','figure_2.py','figure_3.py','figure_4.py','figure_5_extended_3.py','extended_1_2.py','supplementary.py']:
        subprocess.run([sys.executable,str(here/'figures'/script)],env=env,check=True)
    assert len(list(output.glob('*.pdf')))==12
    print(f'Rendered 12 figures to {output}')
if __name__=='__main__':main()
