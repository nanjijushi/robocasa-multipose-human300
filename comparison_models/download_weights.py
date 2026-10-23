"""Download the two comparison weights, verifying SHA256 before installing."""
from pathlib import Path
import argparse,hashlib,json,os,urllib.request

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=root/'weights')
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();m=json.loads((root/'weights_manifest.json').read_text())
    a.output.mkdir(parents=True,exist_ok=True)
    for asset in m['assets']:
        dest=a.output/asset['name']
        if dest.exists() and dest.stat().st_size==asset['bytes'] and digest(dest)==asset['sha256']:
            print('Verified',dest);continue
        if a.verify_only:raise RuntimeError(f'Missing or mismatched weight: {dest}')
        url=f"https://github.com/{m['repository']}/releases/download/{m['release_tag']}/{asset['name']}"
        temp=dest.with_suffix(dest.suffix+'.part')
        print('Downloading',url,flush=True)
        with urllib.request.urlopen(url,timeout=90) as response,temp.open('wb') as f:
            while chunk:=response.read(1024*1024):f.write(chunk)
        if temp.stat().st_size!=asset['bytes'] or digest(temp)!=asset['sha256']:
            raise RuntimeError(f'Integrity failure: {temp}; original file not replaced')
        os.replace(temp,dest);print('Verified',dest)
if __name__=='__main__':main()
