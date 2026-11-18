"""Restore supplied original archives; verify each member against its manifest."""
import argparse,csv,hashlib,io,shutil
from pathlib import Path
from zipfile import ZipFile

def restore(archives,output):
    for path in sorted(archives.glob('*.zip')):
        is_duka=path.name.startswith('Dukascopy_');is_hist=path.name.startswith('HistData_')
        if not (is_duka or is_hist):continue
        with ZipFile(path) as z:
            manifest=list(csv.DictReader(io.StringIO(z.read('manifest.csv').decode())))
            for row in manifest:
                if is_duka:
                    member=f'Dukascopy/{row["date"]}/{row["pair"]}.bi5';target=output/'dukascopy'/row['date']/(row['pair']+'.bi5')
                else:
                    member='HistData/'+row['archive'];target=output/'histdata'/row['archive']
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and hashlib.file_digest(target.open('rb'),'sha256').hexdigest()==row['sha256']:continue
                with z.open(member) as source,target.open('wb') as dest:shutil.copyfileobj(source,dest)
                if hashlib.file_digest(target.open('rb'),'sha256').hexdigest()!=row['sha256']:raise ValueError('Checksum mismatch: '+str(target))
        print('Verified',path.name,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archives',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path('raw'));a=parser.parse_args();restore(a.archives,a.output)
