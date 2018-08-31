from kyoto_preprocess import preprocess_kyoto
from ntc_preprocess import preprocess_ntc
from bccwj_preprocess import preprocess_bccwj
import sys

def main():
    assert len(sys.argv) > 1, 'Usage: python run.py <corpus name(kyoto, ntc, bccwj)>'

    args = sys.argv[1:]
    if args[0] == 'all': args = ['kyoto', 'ntc', 'bccwj']
    for ftype in args:
        if ftype == 'kyoto':
            preprocess_kyoto()
        elif ftype == 'ntc':
            preprocess_ntc()
        elif ftype == 'bccwj':
            preprocess_bccwj()
        else:
            print('<corpus name> must be which any (kyoto, ntc, bccwj)')
            sys.exit()

if __name__ == '__main__':
    main()
